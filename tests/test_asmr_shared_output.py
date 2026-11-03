"""Single physical writers with logical receipts; synthetic audio/inference only."""
from collections import Counter
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from uuid import UUID, uuid4, uuid5

import pytest

from scripts.asmr_transcribe_lib import gui_execution, gui_plan_io
from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.gui_plan import (
    PlanRejected, build_execution_plan, path_key, _hash, _namespace_issues,
)
from scripts.asmr_transcribe_lib.gui_plan_io import encode_plan, decode_plan, PlanTransportError
from scripts.asmr_transcribe_lib.gui_results import (
    RunStore, read_json, reconcile_run, resume_provenance, atomic_json, _digest, _verify_receipt,
)
from scripts.asmr_transcribe_lib.gui_protocol import ProtocolError
from scripts.asmr_transcribe_lib.runtime import CancellationToken, PipelineEvent

SRT = "1\n00:00:00,000 --> 00:00:01,000\nSynthetic subtitle.\n"


def library(tmp_path, formats=("flac", "mp3"), works=("RJ001",), tag="(SEcut)"):
    root = tmp_path / "library"
    for work in works:
        normal = root / work / "01_normal" / "Track01.flac"
        normal.parent.mkdir(parents=True)
        normal.write_bytes(b"SYNTHETIC_REPRESENTATIVE")
        for extension in formats:
            variant = root / work / "02_variant" / ("Track01" + tag + "." + extension)
            variant.parent.mkdir(parents=True, exist_ok=True)
            variant.write_bytes(b"SYNTHETIC_VARIANT_" + extension.encode())
    return root


def execute(plan, directory, monkeypatch, *, fail=False, token=None, cancel_after_copy=False,
            cancel_asr=False, fail_copy=False):
    calls, publishes = Counter(), Counter()
    original = gui_execution._publish_new

    def publish(source, target):
        publishes[path_key(target)] += 1
        if fail_copy and "variant" in str(target):
            raise OSError("synthetic copy owner failure")
        original(source, target)
        if cancel_after_copy and "variant" in str(target):
            token.cancel()

    monkeypatch.setattr(gui_execution, "_publish_new", publish)
    def asr(*, transcribe_srt_path, **kwargs):
        calls["asr"] += 1
        if fail:
            raise RuntimeError("synthetic representative failure")
        if cancel_asr:
            token.cancel()
            token.checkpoint()
        transcribe_srt_path.write_text(SRT, encoding="utf-8")
        return transcribe_srt_path
    def translate(source, target):
        calls["translation"] += 1
        return Path(shutil.copyfile(source, target))
    def convert(source, target):
        calls["opencc"] += 1
        return Path(shutil.copyfile(source, target))
    store = RunStore(directory, str(uuid4()), plan, io.StringIO())
    store.start()
    outcomes = run_plan(plan, project_root=directory.parent, transcriber=asr,
                        translator=translate if plan.mode == "full" else None,
                        opencc_converter=convert, observer=store.observer, cancel_token=token)
    result = store.finish(outcomes)
    return store, outcomes, result, calls, publishes


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
@pytest.mark.parametrize("formats", [("flac", "mp3"), ("flac", "wav", "mp3"), ("wav", "mp3")])
@pytest.mark.parametrize("tag", ["(SEcut)", "(SEなし)"])
def test_unique_writer_and_truthful_receipts(tmp_path, monkeypatch, mode, formats, tag):
    root = library(tmp_path, formats, tag=tag)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.suffix in {".flac", ".wav", ".mp3"}}
    plan = build_execution_plan(root, mode=mode, skip_existing_subtitles=True)
    assert not plan.issues
    assert len(plan.duplicate_groups) == 1
    representative = plan.track_index[plan.duplicate_groups[0].representative_track_id]
    assert representative.task.audio_path.parent.name == "01_normal"
    aliases = [t for t in plan.tracks if t.shared_output_owner]
    assert len(aliases) == len(formats) - 1
    assert all(t.dependencies == (t.shared_output_owner,) for t in aliases)
    role = "transcription" if mode == "transcribe_only" else "final"
    targets = {path_key(t.task.transcribe_srt_path if mode == "transcribe_only" else t.task.final_srt_path)
               for t in plan.tracks}
    for target in targets:
        assert sum(u.access == "write" and u.role == role and path_key(u.path) == target
                   for u in plan.output_ownership) == 1
    replay = decode_plan(encode_plan(plan))
    assert replay == plan and replay.output_ownership == plan.output_ownership
    store, outcomes, result, calls, publishes = execute(replay, tmp_path / "run", monkeypatch)
    assert calls["asr"] == 1
    assert calls["translation"] == (1 if mode == "full" else 0)
    assert len(outcomes) == len(plan.tracks)
    assert Counter(o.status for o in outcomes)["copied"] == len(formats)
    assert result["outcome"] == "completed"
    assert all(count == 1 for count in publishes.values())
    # One publication of the variant output, including transcribe-only mode.
    assert sum(count for key, count in publishes.items() if "variant" in key) == 1
    for track in aliases:
        receipt = read_json(store.run_dir / "receipts" / (track.track_id + ".json"))
        assert receipt["status"] == "copied"
        assert receipt["shared_output_owner"] == track.shared_output_owner
        assert receipt["shared_output_receipt"]["receipt_ref"] == "receipts/" + track.shared_output_owner + ".json"
    recovered = reconcile_run(store.run_dir, store.run_id, plan.plan_hash)
    assert recovered["complete"] and len(recovered["tracks"]) == len(plan.tracks)
    assert len(resume_provenance(store.run_dir, store.run_id, plan.plan_hash)) == len(plan.tracks)
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
@pytest.mark.parametrize("parallel", [1, 5])
def test_owner_after_reader_in_catalog_still_executes_first(tmp_path, monkeypatch, mode, parallel):
    plan = build_execution_plan(library(tmp_path, ("wav", "mp3")), mode=mode, settings={"parallel": parallel})
    reader = next(t for t in plan.tracks if t.shared_output_owner)
    assert reader.ordinal < plan.track_index[reader.shared_output_owner].ordinal
    _, outcomes, result, _, _ = execute(plan, tmp_path / "run", monkeypatch)
    assert result["outcome"] == "completed"
    assert all(o.status in {"success", "transcribed", "copied"} for o in outcomes)


def test_work_boundaries_keep_independent_owners(tmp_path, monkeypatch):
    plan = build_execution_plan(library(tmp_path, works=("RJ001", "RJ002")))
    readers = [t for t in plan.tracks if t.shared_output_owner]
    assert len(readers) == 2
    assert all(plan.track_index[t.shared_output_owner].work_id == t.work_id for t in readers)
    _, _, result, calls, _ = execute(plan, tmp_path / "run", monkeypatch)
    assert result["outcome"] == "completed" and calls["translation"] == 2


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_failed_or_cancelled_representative_never_certifies_shared_output(tmp_path, monkeypatch, mode):
    plan = build_execution_plan(library(tmp_path), mode=mode)
    store, outcomes, result, calls, publishes = execute(plan, tmp_path / "run", monkeypatch, fail=True)
    assert result["outcome"] == "completed_with_errors"
    assert Counter(o.status for o in outcomes) == {"failed": 1, "not_started": 2}
    assert not publishes and calls["translation"] == 0
    assert all(not read_json(p).get("provenance") for p in (store.run_dir / "receipts").glob("*.json"))
    # A cancellation requested before any task leaves every Track not-started.
    monkeypatch.undo()
    token = CancellationToken()
    token.cancel()
    store, outcomes, result, calls, publishes = execute(plan, tmp_path / "cancel", monkeypatch, token=token)
    assert all(o.status == "not_started" for o in outcomes)
    assert not publishes and not calls
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_stop_after_owner_copy_and_restart_with_authoritative_receipts(tmp_path, monkeypatch, mode):
    root = library(tmp_path)
    plan = build_execution_plan(root, mode=mode, skip_existing_subtitles=True)
    store, outcomes, result, calls, publishes = execute(plan, tmp_path / "run", monkeypatch,
                                                        token=CancellationToken(), cancel_after_copy=True)
    reader = next(t for t in plan.tracks if t.shared_output_owner)
    assert next(o for o in outcomes if o.track_id == reader.track_id).status == "not_started"
    assert result["outcome"] == "cancelled"
    proofs = resume_provenance(store.run_dir, store.run_id, plan.plan_hash)
    assert not any(p.track_id == reader.track_id for p in proofs)
    contents = {p: p.read_bytes() for p in root.rglob("*.srt")}
    monkeypatch.undo()
    resumed = build_execution_plan(root, mode=mode, provenance=proofs, skip_existing_subtitles=True)
    assert not resumed.issues
    _, outcomes, result, calls, publishes = execute(resumed, tmp_path / "restart", monkeypatch)
    assert result["outcome"] == "completed" and len(outcomes) == len(plan.tracks)
    assert not calls and not publishes
    assert all(p.read_bytes() == data for p, data in contents.items())


def rehash(plan):
    digest = gui_plan_io._plan_hash(plan)
    return replace(plan, plan_hash=digest, plan_id=str(uuid5(UUID(plan.root_id), "plan/" + digest)))


@pytest.mark.parametrize("change", ["foreign_owner", "self_owner", "remove_owner", "wrong_dependency", "changed_path"])
def test_frozen_plan_rejects_forged_aliases_even_with_new_hash(tmp_path, change):
    plan = build_execution_plan(library(tmp_path, works=("RJ001", "RJ002")))
    reader = next(t for t in plan.tracks if t.shared_output_owner)
    if change == "foreign_owner":
        replacement = replace(reader, shared_output_owner=next(t.track_id for t in plan.tracks if t.work_id != reader.work_id))
    elif change == "self_owner":
        replacement = replace(reader, shared_output_owner=reader.track_id)
    elif change == "remove_owner":
        replacement = replace(reader, shared_output_owner=None)
    elif change == "wrong_dependency":
        replacement = replace(reader, dependencies=(reader.representative_track_id,))
    else:
        replacement = replace(reader, task=replace(reader.task, final_srt_path=reader.task.audio_path.parent / "other.srt"))
    forged = rehash(replace(plan, tracks=tuple(replacement if t.track_id == reader.track_id else t for t in plan.tracks)))
    with pytest.raises(PlanTransportError):
        encode_plan(forged)


def test_unrelated_groups_and_cross_stage_collisions_remain_rejected(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    for extension in ("flac", "mp3"):
        (root / ("Track01." + extension)).write_bytes(b"synthetic")
    plan = build_execution_plan(root)
    assert len(plan.duplicate_groups) == 2  # Root-level legacy no-dedup policy.
    assert not any(t.shared_output_owner for t in plan.tracks)
    with pytest.raises(PlanRejected, match="OUTPUT_NAMESPACE_COLLISION"):
        plan.assert_ready()
    work = root / "Work"
    work.mkdir()
    (work / "voice.wav").write_bytes(b"synthetic")
    (work / "voice-transcribe.wav").write_bytes(b"synthetic")
    plan = build_execution_plan(work)
    assert any(i.code == "OUTPUT_NAMESPACE_COLLISION" for i in plan.issues)


def test_existing_srt_is_read_only_and_unsafe_legacy_intermediate_is_blocked(tmp_path):
    root = library(tmp_path)
    output = root / "RJ001/02_variant/Track01(SEcut).srt"
    output.write_text(SRT, encoding="utf-8")
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    # A pre-existing unreceipted variant may not become a trusted source for
    # the normal Track. Preserve that separate legacy-provenance rejection.
    assert {i.code for i in plan.issues} == {"UNVERIFIED_LEGACY"}
    assert all(t.task.action == "skip" for t in plan.tracks if t.task.audio_path.parent == output.parent)
    assert all(u.access == "read" for u in plan.output_ownership if path_key(u.path) == path_key(output))
    audio = root / "Other/Track99.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"SYNTHETIC_UNRELATED_AUDIO")
    legacy = audio.with_name("Track99-transcribe.srt")
    legacy.write_text(SRT, encoding="utf-8")
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    assert any(legacy in i.paths for i in plan.issues)
    with pytest.raises(PlanRejected, match="UNVERIFIED_LEGACY"):
        plan.assert_ready()
    assert output.read_text(encoding="utf-8") == SRT


def test_shared_receipt_requires_real_owner_receipt(tmp_path, monkeypatch):
    plan = build_execution_plan(library(tmp_path))
    store, _, _, _, _ = execute(plan, tmp_path / "run", monkeypatch)
    reader = next(t for t in plan.tracks if t.shared_output_owner)
    owner_receipt = store.run_dir / "receipts" / (reader.shared_output_owner + ".json")
    # The output still exists, but removing the physical owner's receipt must
    # invalidate the logical reader's completion and resume trust.
    owner_receipt.unlink()
    with pytest.raises((ProtocolError, OSError)):
        resume_provenance(store.run_dir, store.run_id, plan.plan_hash)


def test_old_unaffected_v1_transport_remains_readable(tmp_path):
    plan = build_execution_plan(library(tmp_path, formats=("mp3",)))
    document = encode_plan(plan)
    for track in document["plan"]["tracks"]:
        del track["shared_output_owner"]
    document["transport_hash"] = _hash({k: v for k, v in document.items() if k != "transport_hash"})
    assert decode_plan(document) == plan


@pytest.mark.parametrize("corruption", ["missing_link", "other_receipt", "failed_owner"])
def test_reader_receipt_semantics_checked_independently_of_hash(tmp_path, monkeypatch, corruption):
    plan = build_execution_plan(library(tmp_path))
    store, _, _, _, _ = execute(plan, tmp_path / "run", monkeypatch)
    reader = next(t for t in plan.tracks if t.shared_output_owner)
    path = store.run_dir / "receipts" / (reader.track_id + ".json")
    receipt = read_json(path)
    if corruption == "missing_link":
        del receipt["shared_output_receipt"]
    elif corruption == "other_receipt":
        receipt["shared_output_receipt"]["receipt_ref"] = "receipts/" + reader.representative_track_id + ".json"
    else:
        owner_path = store.run_dir / "receipts" / (reader.shared_output_owner + ".json")
        owner_receipt = read_json(owner_path)
        owner_receipt.update(status="failed", output_hash=None, provenance=None)
        atomic_json(owner_path, owner_receipt)
        receipt["shared_output_receipt"].update(status="failed", receipt_hash=_digest(owner_receipt))
    atomic_json(path, receipt)
    stored = {**store.state["tracks"][reader.track_id], "receipt_hash": _digest(receipt)}
    with pytest.raises(ProtocolError):
        _verify_receipt(store.run_dir, store.manifest, reader.track_id, stored)


def test_reader_cannot_publish_receipt_before_owner(tmp_path):
    plan = build_execution_plan(library(tmp_path))
    reader = next(t for t in plan.tracks if t.shared_output_owner)
    reader.task.final_srt_path.write_text(SRT, encoding="utf-8")
    store = RunStore(tmp_path / "run", str(uuid4()), plan, io.StringIO())
    store.start()
    with pytest.raises(ProtocolError, match="owner lacks"):
        store.observer(PipelineEvent("track_finished", reader.track_id, status="copied"))
    assert not (store.run_dir / "receipts" / (reader.track_id + ".json")).exists()


@pytest.mark.parametrize("status", ["success", "transcribed", "skipped"])
def test_reader_cannot_claim_a_processing_or_skip_receipt(tmp_path, status):
    plan = build_execution_plan(library(tmp_path))
    reader = next(t for t in plan.tracks if t.shared_output_owner)
    store = RunStore(tmp_path / "run", str(uuid4()), plan, io.StringIO())
    store.start()
    with pytest.raises(ProtocolError, match="must report copied"):
        store.observer(PipelineEvent("track_finished", reader.track_id, status=status))
    assert not (store.run_dir / "receipts" / (reader.track_id + ".json")).exists()


def test_execution_never_adds_excluded_dependency_to_scheduler(tmp_path, monkeypatch):
    plan = build_execution_plan(library(tmp_path))
    reader = next(t for t in plan.tracks if t.shared_output_owner)
    malformed = replace(plan, excluded_track_ids=(reader.shared_output_owner,))
    # Simulate an internal caller bypassing transport: ordering must still fail
    # before invoking the scheduler, rather than silently adding an excluded task.
    monkeypatch.setattr(gui_execution, "run_tasks", lambda *args, **kwargs: pytest.fail("scheduler must not run"))
    with pytest.raises(ValueError, match="outside planned execution"):
        run_plan(malformed, project_root=tmp_path, translator=lambda *args: None,
                 transcriber=lambda **kwargs: None, opencc_converter=lambda *args: None)


@pytest.mark.skipif(os.name != "nt", reason="native Windows namespace aliases and leases")
def test_windows_junction_alias_and_cross_instance_ownership(tmp_path, monkeypatch):
    from scripts.asmr_transcribe_lib.gui_ownership import NamespaceLease, OwnershipConflict
    root = library(tmp_path)
    alias = tmp_path / "library_alias"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(root)], check=True, capture_output=True)
    plan = build_execution_plan(root)
    alias_plan = build_execution_plan(alias)
    assert alias_plan == plan
    assert not plan.issues
    assert decode_plan(encode_plan(plan)) == plan
    first = NamespaceLease(tmp_path / "state", str(uuid4()), "Local\\shared-output-test", plan).acquire()
    try:
        with pytest.raises(OwnershipConflict):
            NamespaceLease(tmp_path / "state", str(uuid4()), "Local\\shared-output-test-2", alias_plan).acquire()
    finally:
        first.settle(True)
    _, outcomes, result, _, publishes = execute(plan, tmp_path / "run", monkeypatch)
    assert len(outcomes) == len(plan.tracks) and result["outcome"] == "completed"
    assert all(count == 1 for count in publishes.values())


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
@pytest.mark.parametrize("failure", ["cancel_asr", "fail_copy"])
def test_cancelled_representative_or_failed_copy_owner_never_completes_reader(tmp_path, monkeypatch, mode, failure):
    plan = build_execution_plan(library(tmp_path), mode=mode)
    store, outcomes, result, _, _ = execute(plan, tmp_path / "run", monkeypatch,
                                           token=CancellationToken(), **{failure: True})
    reader = next(t for t in plan.tracks if t.shared_output_owner)
    outcome = next(o for o in outcomes if o.track_id == reader.track_id)
    assert outcome.status == "not_started"
    assert not outcome.provenance
    assert result["outcome"] in {"cancelled", "completed_with_errors"}
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]
