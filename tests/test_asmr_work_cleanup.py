"""All gates: Work-local receipt barriers, native deletion and crash recovery."""
from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
import threading
from uuid import uuid4

import pytest

from scripts.asmr_transcribe_lib import gui_cleanup, pipeline
from scripts.asmr_transcribe_lib.gui_cleanup import IntermediateCleanup, WARNING
from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan, validate_srt, OutputUse
from scripts.asmr_transcribe_lib.gui_results import (
    RunStore, cleanup_completed_work, read_json, reconcile_run, resume_provenance,
)
from scripts.asmr_transcribe_lib.gui_protocol import ProtocolError
from scripts.asmr_transcribe_lib.runtime import CancellationToken
from test_asmr_gui_cleanup import prepare, execute, intermediates, SRT
from test_asmr_gui_srt_normalization import RAW


@pytest.mark.parametrize("mode,keep,cross_reader,stop_b", [
    ("full", False, False, False), ("full", True, False, False),
    ("transcribe_only", False, False, False), ("full", False, True, False),
    ("full", False, False, True),
])
def test_work_a_copies_and_cleanup_do_not_wait_for_work_b(tmp_path, monkeypatch, mode, keep, cross_reader, stop_b):
    root = tmp_path / "library"
    for name in ("RJA/WAV/Track01.flac", "RJA/MP3/Track01.mp3", "RJA/SEなし/Track01.mp3",
                 "RJA/variant/Track01(SEcut).wav", "RJA/variant/Track01(SEcut).mp3", "RJB/Track02.flac"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic audio")
    plan = build_execution_plan(root, mode=mode, settings={"parallel": 5, "transcribe_parallel": 1,
                                                        "keep_intermediate_files": keep})
    a = [t for t in plan.planned_tracks if t.task.work_dir.name == "RJA"]
    b = next(t for t in plan.planned_tracks if t.task.work_dir.name == "RJB")
    owner = next(t for t in a if t.task.action == "process")
    assert len(a) == 5 and sum(t.task.action == "process" for t in a) == 1
    if cross_reader:
        # Future explicit intermediate-reader metadata; D02 still remains Work-local.
        plan.__dict__["output_ownership"] = (*plan.output_ownership, *[
            OutputUse(path, role, b.track_id, owner.track_id, "read")
            for role, path in zip(gui_cleanup.ROLES, intermediates(owner))])
    store = RunStore(tmp_path / "run", str(uuid4()), plan, io.StringIO(), ownership_check=lambda: None)
    store.start()
    a_asr, b_inside, release_b, settled_a = (threading.Event() for _ in range(4))
    remove, correct = pipeline._remove_intermediates, pipeline.apply_default_transcript_corrections
    calls = []
    token = CancellationToken()
    def before_asr(task):
        if task.work_dir.name == "RJB":
            assert a_asr.wait(10)
        remove(task)
    def correction(path, **kwargs):
        if "RJA" in str(path):
            assert b_inside.wait(10)
        return correct(path, **kwargs)
    monkeypatch.setattr(pipeline, "_remove_intermediates", before_asr)
    monkeypatch.setattr(pipeline, "apply_default_transcript_corrections", correction)
    def asr(*, audio_path, transcribe_srt_path, **kwargs):
        calls.append(audio_path)
        if audio_path == b.task.audio_path:
            b_inside.set()
            assert release_b.wait(20)
        transcribe_srt_path.write_bytes(RAW if audio_path == owner.task.audio_path else SRT)
        if audio_path == owner.task.audio_path:
            a_asr.set()
        return transcribe_srt_path
    def copy(source, target):
        target.write_bytes(source.read_bytes())
        return target
    def observe(event):
        store.observer(event)
        if event.status in {"success", "transcribed", "copied"} and all(
                store.state["tracks"][t.track_id]["status"] in {"success", "transcribed", "copied"} for t in a):
            settled_a.set()
    def run():
        outcomes = run_plan(plan, project_root=tmp_path, transcriber=asr,
                            translator=copy if mode == "full" else None, opencc_converter=copy,
                            observer=observe, cancel_token=token, normalization_store=store.record_normalization,
                            intermediate_store=store.intermediate_cleanup.capture)
        return outcomes, store.finish(outcomes)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run)
        try:
            assert b_inside.wait(10) and settled_a.wait(10)
            assert not future.done() and not store.finished and not release_b.is_set()
            recovered = reconcile_run(store.run_dir, store.run_id, plan.plan_hash)
            assert not recovered["complete"]
            for track in a:
                receipt = read_json(store.run_dir / recovered["tracks"][track.track_id]["receipt_ref"])
                assert receipt["provenance"]
                validate_srt(Path(receipt["output_key"]))
            checkpoint = read_json(store.run_dir / "intermediate_cleanup.json")["works"][owner.work_id]
            if mode == "transcribe_only":
                assert owner.task.transcribe_srt_path.exists() and checkpoint["tracks"] == {}
            else:
                assert all(p.exists() == (keep or cross_reader) for p in intermediates(owner))
                assert checkpoint["status"] == ("deferred_dependencies" if cross_reader else "completed")
                assert checkpoint["tracks"][owner.track_id]["status"] == (
                    "disabled" if keep else "deferred_dependencies" if cross_reader else "completed")
            evidence = read_json(store.run_dir / "receipts" / (owner.track_id + ".json"))["timestamp_normalization"]
            audit = read_json(store.run_dir / evidence["ref"])
            assert audit["repaired_cues"] == 1 and audit["max_displacement_ms"] == 420
            assert (store.run_dir / audit["raw_ref"]).read_bytes() == RAW
        finally:
            if stop_b:
                token.cancel()
            release_b.set()
        outcomes, result = future.result(timeout=15)
    assert len(calls) == 2 and len(outcomes) == 6 and result["outcome"] == ("cancelled" if stop_b else "completed")
    assert result["counts"]["copied"] == 4
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]
    if mode == "full" and not keep:
        assert all(not p.exists() for p in intermediates(owner))
    log = (store.run_dir / "logs/execution.log").read_text(encoding="utf-8")
    assert log.count("作品結算") == 2 and "SRT_TIMESTAMP_NORMALIZED" in log and "Synthetic" not in log


def test_mixed_failure_retains_failed_track_and_cleans_independent_success(tmp_path):
    plan, store = prepare(tmp_path, names=("work/Track01_intro.wav", "work/Track02_epilogue.wav"))
    def asr(*, transcribe_srt_path, **kwargs):
        transcribe_srt_path.write_bytes(SRT)
        return transcribe_srt_path
    def translate(source, target):
        target.write_bytes(source.read_bytes())
        if "epilogue" in source.name:
            raise RuntimeError("PRIVATE_CAPTION token=PRIVATE_SECRET")
        return target
    outcomes = run_plan(plan, project_root=tmp_path, transcriber=asr, translator=translate,
                        opencc_converter=translate, observer=store.observer,
                        intermediate_store=store.intermediate_cleanup.capture)
    result = store.finish(outcomes)
    good = next(t for t in plan.tracks if "intro" in t.task.audio_path.name)
    bad = next(t for t in plan.tracks if "epilogue" in t.task.audio_path.name)
    assert result["outcome"] == "completed_with_errors" and result["counts"]["failed"] == 1
    assert all(not p.exists() for p in intermediates(good))
    assert bad.task.transcribe_srt_path.exists() and bad.task.translate_srt_path.exists()
    assert bad.track_id not in result["intermediate_cleanup"]
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]
    log = (store.run_dir / "logs/execution.log").read_text(encoding="utf-8")
    assert log.count("作品結算") == 1 and "含失敗" in log and "PRIVATE_" not in log


def test_copy_and_skip_only_work_has_no_cleanup_candidates(tmp_path):
    plan, store = prepare(tmp_path, names=("work/WAV/voice.wav",))
    execute(plan, store)
    variant = plan.root / "work/MP3/voice.mp3"
    variant.parent.mkdir()
    variant.write_bytes(b"synthetic variant")
    resumed = build_execution_plan(plan.root, provenance=resume_provenance(store.run_dir, store.run_id, plan.plan_hash))
    assert resumed.executable and not any(t.task.action == "process" for t in resumed.planned_tracks)
    next_store = RunStore(tmp_path / "restart", str(uuid4()), resumed, io.StringIO(), ownership_check=lambda: None)
    next_store.start()
    def forbidden(**kwargs):
        raise AssertionError("copy-only Work invoked inference")
    outcomes = run_plan(resumed, project_root=tmp_path, transcriber=forbidden, translator=forbidden,
                        opencc_converter=forbidden, observer=next_store.observer)
    result = next_store.finish(outcomes)
    assert result["intermediate_cleanup"] == {} and result["warnings"] == []
    assert result["counts"]["copied"] == result["counts"]["skipped"] == 1
    assert reconcile_run(next_store.run_dir, next_store.run_id, resumed.plan_hash)["complete"]


class SimulatedCrash(BaseException):
    pass


@pytest.mark.parametrize("point", ["before_checkpoint", "before_delete", "after_delete", "partial"])
def test_crash_checkpoint_and_fresh_cleanup_engine_recovery(tmp_path, monkeypatch, point):
    import scripts.asmr_transcribe_lib.gui_results as records
    plan, store = prepare(tmp_path)
    remove, atomic = gui_cleanup._remove_owned, records.atomic_json
    calls = []
    def crash_delete(*args):
        calls.append(args[0])
        if point == "before_delete" or point == "partial" and len(calls) == 2:
            raise SimulatedCrash()
        remove(*args)
        if point == "after_delete":
            raise SimulatedCrash()
    def crash_checkpoint(path, value):
        if point == "before_checkpoint" and path.name == "intermediate_cleanup.json":
            raise SimulatedCrash()
        return atomic(path, value)
    monkeypatch.setattr(gui_cleanup, "_remove_owned", crash_delete)
    monkeypatch.setattr(records, "atomic_json", crash_checkpoint)
    try:
        with pytest.raises(SimulatedCrash):
            execute(plan, store, normalized=True)
        before = reconcile_run(store.run_dir, store.run_id, plan.plan_hash)
        assert not before["complete"] and before["counts"]["success"] == 1
        if point != "before_checkpoint":
            assert WARNING in before["warnings"]
        monkeypatch.setattr(gui_cleanup, "_remove_owned", remove)
        monkeypatch.setattr(records, "atomic_json", atomic)
        engine = IntermediateCleanup(plan, lambda: None)
        assert not engine.stamps  # Recovery uses receipt-bound evidence, not fresh adoption.
        repaired = cleanup_completed_work(store.run_dir, store.run_id, plan.works[0].work_id, engine, retry=True)
        assert repaired["status"] == "completed"
        assert all(not p.exists() for p in intermediates(plan.tracks[0]))
        if point == "after_delete":
            assert "absent_after_intent" in repaired["tracks"][plan.tracks[0].track_id]["files"].values()
        assert cleanup_completed_work(store.run_dir, store.run_id, plan.works[0].work_id, engine, retry=True) == repaired
        proof = resume_provenance(store.run_dir, store.run_id, plan.plan_hash)
        resumed = build_execution_plan(plan.root, provenance=proof)
        assert resumed.executable and not any(t.task.action == "process" for t in resumed.planned_tracks)
        receipt = read_json(store.run_dir / "receipts" / (plan.tracks[0].track_id + ".json"))
        audit = read_json(store.run_dir / receipt["timestamp_normalization"]["ref"])
        assert (store.run_dir / audit["raw_ref"]).read_bytes() == RAW
    finally:
        store._readable.close(wait=True)


def test_cleanup_retry_limit_and_namespace_loss_are_fail_closed(tmp_path, monkeypatch):
    plan, store = prepare(tmp_path)
    calls = []
    def locked(*args):
        calls.append(args[0])
        raise PermissionError("controlled sharing conflict")
    monkeypatch.setattr(gui_cleanup, "_remove_owned", locked)
    _, result = execute(plan, store)
    assert result["outcome"] == "completed" and result["warnings"] == [WARNING]
    engine = IntermediateCleanup(plan, lambda: None)
    for _ in range(5):
        last = cleanup_completed_work(store.run_dir, store.run_id, plan.works[0].work_id, engine, retry=True)
    assert last["attempts"] == 3 and len(calls) == 9
    assert all(p.exists() for p in intermediates(plan.tracks[0]))
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]


def test_hardlink_added_after_validation_is_retained(tmp_path):
    plan, store = prepare(tmp_path)
    alias = tmp_path / "external.srt"
    def link(results):
        os.link(plan.tracks[0].task.transcribe_srt_path, alias)
    _, result = execute(plan, store, before_finish=link)
    assert plan.tracks[0].task.transcribe_srt_path.exists() and alias.read_bytes() == SRT
    assert result["outcome"] == "completed" and result["warnings"] == [WARNING]


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows file sharing")
def test_windows_locked_intermediate_is_nonfatal_and_retryable(tmp_path):
    import ctypes
    from scripts.asmr_transcribe_lib.gui_ownership import _kernel
    plan, store = prepare(tmp_path)
    kernel, handles = _kernel(), []
    locked = plan.tracks[0].task.translate_srt_path
    def lock(results):
        handle = kernel.CreateFileW(str(locked), 0x80000000, 3, None, 3, 0x80, None)
        assert handle != ctypes.c_void_p(-1).value
        handles.append(handle)
    try:
        _, result = execute(plan, store, before_finish=lock)
        assert result["outcome"] == "completed" and result["warnings"] == [WARNING] and locked.exists()
        assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]
    finally:
        for handle in handles:
            kernel.CloseHandle(handle)
    done = cleanup_completed_work(store.run_dir, store.run_id, plan.works[0].work_id,
                                  IntermediateCleanup(plan, lambda: None), retry=True)
    assert done["status"] == "completed" and not locked.exists()


@pytest.mark.parametrize("after_delete", [False, True])
def test_cleanup_checkpoint_io_failure_is_nonfatal_and_recoverable(tmp_path, monkeypatch, after_delete):
    import scripts.asmr_transcribe_lib.gui_results as records
    plan, store = prepare(tmp_path)
    atomic = records.atomic_json
    def fail(path, value):
        if path.name == "intermediate_cleanup.json":
            removed = any("removed" in t["files"].values() for w in value["works"].values() for t in w["tracks"].values())
            if not after_delete or removed:
                raise OSError("PRIVATE_SECRET controlled metadata write failure")
        return atomic(path, value)
    monkeypatch.setattr(records, "atomic_json", fail)
    _, result = execute(plan, store)
    assert result["outcome"] == "completed" and result["warnings"] == [WARNING]
    assert plan.tracks[0].task.final_srt_path.exists()
    assert sum(p.exists() for p in intermediates(plan.tracks[0])) == (2 if after_delete else 3)
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]
    monkeypatch.setattr(records, "atomic_json", atomic)
    done = cleanup_completed_work(store.run_dir, store.run_id, plan.works[0].work_id,
                                  IntermediateCleanup(plan, lambda: None), retry=True)
    assert done["status"] == "completed" and all(not p.exists() for p in intermediates(plan.tracks[0]))


def test_final_hash_is_verified_before_work_cleanup(tmp_path):
    from scripts.asmr_transcribe_lib.runtime import ObserverError
    plan, store = prepare(tmp_path)
    def corrupt(results):
        plan.tracks[0].task.final_srt_path.write_bytes(SRT.replace(b"synthetic", b"changed"))
    try:
        with pytest.raises(ObserverError) as error:
            execute(plan, store, before_finish=corrupt)
        assert isinstance(error.value.__cause__, ProtocolError)
        assert all(p.exists() for p in intermediates(plan.tracks[0]))
    finally:
        store._readable.close(wait=True)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows file sharing")
def test_windows_pinned_file_rejects_rename_after_validation(tmp_path):
    plan, store = prepare(tmp_path)
    target = plan.tracks[0].task.transcribe_srt_path
    calls, blocked = [], []
    def ownership():
        calls.append(True)
        if len(calls) == 2:
            try:
                target.rename(tmp_path / "replacement.srt")
            except OSError:
                blocked.append(True)
            else:
                raise AssertionError("validated file could be replaced while pinned")
    store.intermediate_cleanup.ownership_check = ownership
    _, result = execute(plan, store)
    assert blocked and result["outcome"] == "completed" and not result["warnings"]
    assert all(not p.exists() for p in intermediates(plan.tracks[0]))
