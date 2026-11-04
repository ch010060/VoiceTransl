"""Deterministic post-receipt lifecycle and exact-path cleanup regression."""
import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from scripts.asmr_transcribe_lib.gui_cleanup import ROLES, WARNING
from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan, validate_srt
from scripts.asmr_transcribe_lib.gui_results import RunStore, read_json, reconcile_run, safe_ref
from scripts.asmr_transcribe_lib.gui_protocol import ProtocolError
from scripts.asmr_transcribe_lib.runtime import CancellationToken, ObserverError
from test_asmr_gui_srt_normalization import RAW

SRT = b"1\n00:00:00,000 --> 00:00:01,000\nsynthetic subtitle\n\n"


def prepare(tmp_path, *, keep=False, mode="full", names=("voice.wav",), existing=False):
    root = tmp_path / "library"
    root.mkdir()
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic audio")
        if existing:
            path.with_suffix(".srt").write_bytes(SRT)
            for suffix in ("-transcribe.srt", "-translate.srt", "-opencc.srt"):
                path.with_name(path.stem + suffix).write_bytes(b"unrelated")
    plan = build_execution_plan(root, mode=mode, settings={"keep_intermediate_files": keep})
    assert plan.executable
    store = RunStore(tmp_path / "run", str(uuid4()), plan, io.StringIO(), ownership_check=lambda: None)
    store.start()
    return plan, store


def execute(plan, store, *, fail=None, normalized=False, token=None, before_finish=None):
    if before_finish:
        cleanup = store.cleanup_work
        called = []
        def before_cleanup(work_id):
            if not called and all(t["status"] in {"success", "failed", "copied", "skipped", "transcribed", "cancelled", "not_started"}
                                  for t in store.state["tracks"].values()):
                called.append(work_id)
                outcomes = [SimpleNamespace(track_id=key, status=t["status"],
                            output_path=Path(store.identities[key]["output_key"])) for key, t in store.state["tracks"].items()]
                before_finish(outcomes)
                store._save_state()
            return cleanup(work_id)
        store.cleanup_work = before_cleanup
    def asr(*, transcribe_srt_path, **kwargs):
        transcribe_srt_path.write_bytes(RAW if normalized else SRT)
        if fail == "asr":
            raise RuntimeError("synthetic failure")
        if token is not None:
            token.cancel()
        return transcribe_srt_path
    def stage(name):
        def convert(source, target):
            target.write_bytes(source.read_bytes())
            if fail == name:
                raise RuntimeError("synthetic failure")
            return target
        return convert
    results = run_plan(plan, project_root=plan.root, transcriber=asr,
                       translator=stage("translation") if plan.mode == "full" else None, opencc_converter=stage("opencc"),
                       observer=store.observer, cancel_token=token,
                       normalization_store=store.record_normalization,
                       intermediate_store=store.intermediate_cleanup.capture)
    return results, store.finish(results)


def intermediates(track):
    return [track.task.transcribe_srt_path, track.task.translate_srt_path, track.task.opencc_srt_path]


def recovered(store):
    return reconcile_run(store.run_dir, store.run_id, store.plan.plan_hash)


@pytest.mark.parametrize("keep", [False, True])
def test_success_cleanup_and_keep(tmp_path, keep):
    plan, store = prepare(tmp_path, keep=keep)
    def barrier(results):
        assert all(p.exists() for p in intermediates(plan.tracks[0]))
        assert (store.run_dir / "receipts" / (results[0].track_id + ".json")).exists()
        assert recovered(store)["tracks"][results[0].track_id]["status"] == "success"
    results, result = execute(plan, store, before_finish=barrier)
    assert result["outcome"] == "completed"
    assert all(p.exists() == keep for p in intermediates(plan.tracks[0]))
    validate_srt(results[0].output_path)
    assert result["intermediate_cleanup"][results[0].track_id]["status"] == ("disabled" if keep else "completed")
    assert recovered(store)["complete"]
    assert store.finish(results) == result


def test_transcribe_only_final_never_deleted(tmp_path):
    plan, store = prepare(tmp_path, mode="transcribe_only")
    results, result = execute(plan, store)
    assert results[0].status == "transcribed"
    assert plan.tracks[0].task.transcribe_srt_path.exists()
    assert not plan.tracks[0].task.final_srt_path.exists()
    assert result["intermediate_cleanup"] == {}
    assert recovered(store)["complete"]


@pytest.mark.parametrize("stage", ["asr", "translation", "opencc"])
def test_failed_stages_retain_artifacts(tmp_path, stage):
    plan, store = prepare(tmp_path)
    results, result = execute(plan, store, fail=stage)
    assert results[0].status == "failed"
    assert plan.tracks[0].task.transcribe_srt_path.exists()
    if stage in {"translation", "opencc"}:
        assert plan.tracks[0].task.translate_srt_path.exists()
    if stage == "opencc":
        assert plan.tracks[0].task.opencc_srt_path.exists()
    assert result["intermediate_cleanup"] == {}


def test_cancel_retains_artifacts(tmp_path):
    plan, store = prepare(tmp_path)
    results, result = execute(plan, store, token=CancellationToken())
    assert results[0].status == "cancelled"
    assert plan.tracks[0].task.transcribe_srt_path.exists()
    assert result["intermediate_cleanup"] == {}


def test_skip_does_not_delete_unowned_files(tmp_path):
    plan, store = prepare(tmp_path, existing=True)
    results, result = execute(plan, store)
    assert results[0].status == "skipped"
    assert all(p.read_bytes() == b"unrelated" for p in intermediates(plan.tracks[0]))
    assert result["intermediate_cleanup"] == {}


@pytest.mark.parametrize("names", [
    ("work/normal/Track01.wav", "work/variant/Track01(SEcut).wav"),
    ("work/normal/Track01.wav", "work/variant/Track01(SEcut).wav", "work/variant/Track01(SEcut).flac"),
])
def test_duplicate_and_shared_readers_finish_before_cleanup(tmp_path, names):
    plan, store = prepare(tmp_path, names=names)
    def barrier(results):
        assert len(results) == len(names)
        for track in plan.tracks:
            assert track.task.final_srt_path.exists()
    results, result = execute(plan, store, before_finish=barrier)
    assert any(r.status == "copied" for r in results)
    assert all(t.task.final_srt_path.exists() for t in plan.tracks)
    assert recovered(store)["complete"]
    for track in plan.tracks:
        if track.task.action != "process" or track.shared_output_owner:
            assert track.track_id not in result["intermediate_cleanup"]


def test_normalized_evidence_survives_cleanup_and_tamper_fails(tmp_path):
    plan, store = prepare(tmp_path)
    results, result = execute(plan, store, normalized=True)
    track_id = results[0].track_id
    receipt = read_json(store.run_dir / "receipts" / (track_id + ".json"))
    audit_path = safe_ref(store.run_dir, receipt["timestamp_normalization"]["ref"])
    audit = read_json(audit_path)
    assert safe_ref(store.run_dir, audit["raw_ref"]).read_bytes() == RAW
    assert safe_ref(store.run_dir, audit["normalized_ref"]).exists()
    assert all(not p.exists() for p in intermediates(plan.tracks[0]))
    assert recovered(store)["complete"]
    audit_path.write_text("{}")
    with pytest.raises(ProtocolError):
        recovered(store)


@pytest.mark.parametrize("corrupt", [True, False])
def test_final_corruption_or_missing_still_fails(tmp_path, corrupt):
    plan, store = prepare(tmp_path)
    results, _ = execute(plan, store)
    final = results[0].output_path
    if corrupt:
        final.write_bytes(SRT.replace(b"synthetic", b"corrupted"))
    else:
        final.unlink()
    with pytest.raises((ProtocolError, OSError)):
        recovered(store)


@pytest.mark.parametrize("substitution", ["symlink", "hardlink", "regular", "directory"])
def test_substitution_never_deletes_replacement_or_external_target(tmp_path, substitution):
    plan, store = prepare(tmp_path)
    outside = tmp_path / "external"
    outside.mkdir()
    external = outside / "external.srt"
    external.write_bytes(SRT)
    path = intermediates(plan.tracks[0])[0]
    def replace_path(results):
        path.unlink()
        if substitution == "symlink":
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(path), str(outside)], check=True, capture_output=True)
            else:
                path.symlink_to(external)
        elif substitution == "hardlink":
            os.link(external, path)
        elif substitution == "regular":
            path.write_bytes(b"unrelated replacement")
        else:
            path.mkdir()
    _, result = execute(plan, store, before_finish=replace_path)
    assert path.exists() and external.read_bytes() == SRT
    assert result["warnings"] == [WARNING]
    assert recovered(store)["complete"]


def test_parent_substitution_is_rejected(tmp_path):
    plan, store = prepare(tmp_path, names=("work/voice.wav",))
    def replace_parent(results):
        parent = plan.tracks[0].task.audio_path.parent
        moved = parent.with_name("moved")
        parent.rename(moved)
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(parent), str(moved)], check=True, capture_output=True)
        else:
            parent.symlink_to(moved, target_is_directory=True)
    # The earlier receipt barrier itself rejects changed source identity.
    with pytest.raises(ObserverError) as caught:
        execute(plan, store, before_finish=replace_parent)
    assert isinstance(caught.value.__cause__, ProtocolError)
    assert all(p.exists() for p in intermediates(plan.tracks[0]))


def test_locked_file_keeps_success_and_warning(tmp_path, monkeypatch):
    import scripts.asmr_transcribe_lib.gui_cleanup as module
    plan, store = prepare(tmp_path)
    locked = intermediates(plan.tracks[0])[1]
    original = module._remove_owned
    def unlink(path, *args, **kwargs):
        if path == locked:
            raise PermissionError("private synthetic path must not appear in result")
        return original(path, *args, **kwargs)
    def lock(results):
        monkeypatch.setattr(module, "_remove_owned", unlink)
    results, result = execute(plan, store, before_finish=lock)
    assert results[0].status == "success" and result["outcome"] == "completed"
    assert locked.exists()
    assert result["warnings"] == [WARNING]
    assert "private synthetic" not in json.dumps(result)
    assert recovered(store)["warnings"] == [WARNING]


def test_receipt_failure_prevents_any_cleanup(tmp_path, monkeypatch):
    plan, store = prepare(tmp_path)
    import scripts.asmr_transcribe_lib.gui_results as module
    original = module.atomic_json
    def write(path, payload):
        if path.parent.name == "receipts":
            raise OSError("synthetic durable receipt failure")
        return original(path, payload)
    monkeypatch.setattr(module, "atomic_json", write)
    with pytest.raises(Exception):
        execute(plan, store)
    assert all(p.exists() for p in intermediates(plan.tracks[0]))


def test_namespace_loss_defers_cleanup(tmp_path):
    plan, store = prepare(tmp_path)
    def reject():
        raise ValueError("namespace lost")
    store.intermediate_cleanup.ownership_check = reject
    _, result = execute(plan, store)
    assert all(p.exists() for p in intermediates(plan.tracks[0]))
    assert result["warnings"] == [WARNING]
    assert recovered(store)["complete"]


def test_dependency_read_is_retained(tmp_path):
    plan, store = prepare(tmp_path, names=("work/normal/Track01.wav", "work/variant/Track01(SEcut).wav"))
    from scripts.asmr_transcribe_lib.gui_plan import OutputUse
    owner = next(t for t in plan.tracks if t.task.action == "process")
    reader = next(t for t in plan.tracks if t.track_id != owner.track_id)
    # An intermediate reader is protected until its authoritative receipt settles.
    plan.__dict__["output_ownership"] = (*plan.output_ownership, *[
        OutputUse(path, role, reader.track_id, owner.track_id, "read")
        for role, path in zip(ROLES, intermediates(owner))])
    _, result = execute(plan, store)
    assert all(not p.exists() for p in intermediates(owner))
    assert result["intermediate_cleanup"][owner.track_id]["status"] == "completed"


def test_frozen_setting_and_malformed_values(tmp_path):
    plan, store = prepare(tmp_path, keep=True)
    settings = plan.settings
    settings["keep_intermediate_files"] = False
    assert plan.settings["keep_intermediate_files"] is True
    for value in [None, 0, 1, "false", [], {}]:
        with pytest.raises(ValueError, match="boolean"):
            build_execution_plan(plan.root, settings={"keep_intermediate_files": value})


def test_cleanup_does_not_run_before_all_receipts_verified(tmp_path):
    plan, store = prepare(tmp_path)
    def corrupt_receipt(results):
        path = store.run_dir / "receipts" / (results[0].track_id + ".json")
        path.write_text("{}")
    with pytest.raises(ObserverError) as caught:
        execute(plan, store, before_finish=corrupt_receipt)
    assert isinstance(caught.value.__cause__, ProtocolError)
    assert all(p.exists() for p in intermediates(plan.tracks[0]))


def test_cancel_after_success_preserves_intermediates(tmp_path):
    plan, store = prepare(tmp_path)
    def cancel(results):
        store.state["cancel_requested"] = True
    _, result = execute(plan, store, before_finish=cancel)
    assert all(p.exists() for p in intermediates(plan.tracks[0]))
    assert next(iter(result["intermediate_cleanup"].values()))["status"] == "deferred_interrupted"


def test_cleanup_without_namespace_authority_retains_files(tmp_path):
    plan, store = prepare(tmp_path)
    store.intermediate_cleanup.ownership_check = None
    _, result = execute(plan, store)
    assert all(p.exists() for p in intermediates(plan.tracks[0]))
    assert result["warnings"] == [WARNING]


def test_frozen_transport_rejects_malformed_cleanup_setting(tmp_path):
    from dataclasses import replace
    from scripts.asmr_transcribe_lib.gui_plan import _hash
    from scripts.asmr_transcribe_lib.gui_plan_io import _validate, PlanTransportError
    plan, _ = prepare(tmp_path)
    settings = {**plan.settings, "keep_intermediate_files": "false"}
    malformed = replace(plan, settings_json=json.dumps(settings), settings_hash=_hash(settings))
    with pytest.raises(PlanTransportError, match="cleanup preference"):
        _validate(malformed)


def test_legacy_cli_has_no_cleanup_option():
    from scripts.asmr_transcribe_lib.cli import build_parser
    assert "keep_intermediate_files" not in vars(build_parser().parse_args(["--path", "."]))


def test_final_path_is_protected_even_if_cleanup_input_is_wrong(tmp_path, monkeypatch):
    import scripts.asmr_transcribe_lib.gui_cleanup as module
    plan, store = prepare(tmp_path)
    def alias_final(results):
        original = module._paths
        monkeypatch.setattr(module, "_paths", lambda task: (task.final_srt_path, *original(task)[1:]))
    results, result = execute(plan, store, before_finish=alias_final)
    assert results[0].output_path.read_bytes() == SRT
    assert result["warnings"] == [WARNING]
    assert recovered(store)["complete"]


def test_running_worker_rechecks_initial_leases_not_new_output_inodes(tmp_path):
    from scripts.asmr_transcribe_lib.gui_ownership import (
        NamespaceLease, OwnershipConflict, verify_worker_ownership,
    )
    from scripts.asmr_transcribe_lib.gui_results import atomic_json
    plan, store = prepare(tmp_path)
    state = tmp_path / "namespace"
    lease = NamespaceLease(state, store.run_id, "synthetic-job", plan)
    for resource, exclusive in lease.resources:
        atomic_json(state / "ownership" / resource / (store.run_id + ".json"), {
            "schema_version": "1.0", "run_id": store.run_id, "job_name": "synthetic-job",
            "plan_hash": plan.plan_hash, "resource_id": resource,
            "access": "write" if exclusive else "read", "state": "unsettled"})
    resources = verify_worker_ownership(state, store.run_id, "synthetic-job", plan)
    store.intermediate_cleanup.ownership_check = lambda: verify_worker_ownership(
        state, store.run_id, "synthetic-job", plan, resources=resources)
    _, result = execute(plan, store)
    assert next(iter(result["intermediate_cleanup"].values()))["status"] == "completed"
    assert recovered(store)["complete"]
    marker = state / "ownership" / resources[0][0] / (store.run_id + ".json")
    marker.write_text("{}")
    with pytest.raises(OwnershipConflict):
        verify_worker_ownership(state, store.run_id, "synthetic-job", plan, resources=resources)
    marker.unlink()
    with pytest.raises(OwnershipConflict):
        verify_worker_ownership(state, store.run_id, "synthetic-job", plan, resources=resources)
