"""Dependency-ready scheduling: synthetic inference and explicit barriers."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import io
from pathlib import Path
import shutil
import threading
from uuid import uuid4

import pytest

from scripts.asmr_transcribe_lib import pipeline
from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan
from scripts.asmr_transcribe_lib.gui_results import RunStore, read_json, reconcile_run, resume_provenance
from scripts.asmr_transcribe_lib.runtime import CancellationToken

SRT = "1\n00:00:00,000 --> 00:00:01,000\nPRIVATE_CAPTION_DO_NOT_LOG\n"


def library(tmp_path, works=("RJ001",), variants=True):
    root = tmp_path / "library"
    for work in works:
        for extension in (("flac", "mp3") if variants else ("flac",)):
            path = root / work / ("voice01." + extension)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"SYNTHETIC_AUDIO")
    return root


def execute(plan, directory, *, asr=None, translate=None, convert=None, observer=None, token=None):
    def transcription(*, transcribe_srt_path, **kwargs):
        transcribe_srt_path.write_text(SRT, encoding="utf-8")
        return transcribe_srt_path
    def copy(source, target):
        return Path(shutil.copyfile(source, target))
    store = RunStore(directory, str(uuid4()), plan, io.StringIO())
    store.start()
    def observe(event):
        store.observer(event)
        if observer:
            observer(event, store)
    outcomes = run_plan(plan, project_root=directory.parent, transcriber=asr or transcription,
                        translator=(translate or copy) if plan.mode == "full" else None,
                        opencc_converter=convert or copy, observer=observe, cancel_token=token)
    result = store.finish(outcomes)
    return store, outcomes, result


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_ready_copy_receipt_finishes_while_other_work_asr_is_blocked(tmp_path, monkeypatch, mode):
    plan = build_execution_plan(library(tmp_path, works=("RJ001", "RJ002")), mode=mode,
                                settings={"parallel": 5, "transcribe_parallel": 1})
    b_inside, release_b, copy_receipt, a_asr_finished = (threading.Event() for _ in range(4))
    original = pipeline.apply_default_transcript_corrections
    remove = pipeline._remove_intermediates
    def before_asr(task):
        if task.work_dir.name == "RJ002":
            assert a_asr_finished.wait(10)
        remove(task)
    monkeypatch.setattr(pipeline, "_remove_intermediates", before_asr)
    def correction(path, **kwargs):
        if "RJ001" in str(path):
            assert b_inside.wait(10)
        return original(path, **kwargs)
    monkeypatch.setattr(pipeline, "apply_default_transcript_corrections", correction)
    calls = Counter()
    def asr(*, audio_path, transcribe_srt_path, **kwargs):
        calls[audio_path.parent.name] += 1
        if audio_path.parent.name == "RJ002":
            b_inside.set()
            assert release_b.wait(20)
        transcribe_srt_path.write_text(SRT, encoding="utf-8")
        if audio_path.parent.name == "RJ001":
            a_asr_finished.set()
        return transcribe_srt_path
    def observe(event, store):
        if event.status == "copied" and plan.track_index[event.track_id].task.work_dir.name == "RJ001":
            receipt = read_json(store.run_dir / "receipts" / (event.track_id + ".json"))
            assert receipt["provenance"] and not release_b.is_set()
            copy_receipt.set()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(execute, plan, tmp_path / "run", asr=asr, observer=observe)
        try:
            assert b_inside.wait(10)
            assert copy_receipt.wait(10), "A copy waited for unrelated B ASR"
            assert not future.done()
        finally:
            release_b.set()
        store, outcomes, result = future.result(timeout=15)
    assert result["outcome"] == "completed" and len(outcomes) == 4
    assert calls == {"RJ001": 1, "RJ002": 1}
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]


@pytest.mark.parametrize("parallel", [1, 5])
def test_ready_engine_bounds_active_futures_and_prioritizes_dependencies(tmp_path, monkeypatch, parallel):
    root = library(tmp_path, works=tuple(f"RJ{i:03}" for i in range(8)))
    plan = build_execution_plan(root, settings={"parallel": parallel})
    original = pipeline.ThreadPoolExecutor
    maximum = [0]
    class CheckedExecutor(original):
        def submit(self, *args, **kwargs):
            maximum[0] = max(maximum[0], len([f for f in getattr(self, "submitted", ()) if not f.done()]) + 1)
            result = super().submit(*args, **kwargs)
            self.submitted = [*getattr(self, "submitted", ()), result]
            return result
    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", CheckedExecutor)
    store, outcomes, result = execute(plan, tmp_path / "run")
    assert maximum[0] <= parallel and result["outcome"] == "completed"
    assert len(outcomes) == len({o.track_id for o in outcomes}) == 16


def test_parallel_one_settles_work_before_unrelated_work_starts(tmp_path):
    plan = build_execution_plan(library(tmp_path, works=("RJ001", "RJ002")), settings={"parallel": 1})
    copied_a = threading.Event()
    def asr(*, audio_path, transcribe_srt_path, **kwargs):
        if audio_path.parent.name == "RJ002":
            assert copied_a.is_set()
        transcribe_srt_path.write_text(SRT, encoding="utf-8")
        return transcribe_srt_path
    def observe(event, store):
        if event.status == "copied" and plan.track_index[event.track_id].task.work_dir.name == "RJ001":
            copied_a.set()
    _, _, result = execute(plan, tmp_path / "run", asr=asr, observer=observe)
    assert result["outcome"] == "completed" and copied_a.is_set()


def test_copy_only_restart_uses_authoritative_source_without_new_inference(tmp_path):
    root = library(tmp_path, variants=False)
    variant = root / "RJ001/variant/voice01.mp3"
    variant.parent.mkdir()
    variant.write_bytes(b"SYNTHETIC_VARIANT")
    plan = build_execution_plan(root)
    token = CancellationToken()
    def stop_after_representative(event, store):
        if event.status == "success":
            token.cancel()
    store, _, _ = execute(plan, tmp_path / "run", token=token, observer=stop_after_representative)
    proofs = resume_provenance(store.run_dir, store.run_id, plan.plan_hash)
    resumed = build_execution_plan(root, provenance=proofs)
    assert not resumed.issues
    assert not any(t.task.action == "process" for t in resumed.planned_tracks)
    def forbidden(**kwargs):
        raise AssertionError("copy-only Work must not invoke ASR")
    _, outcomes, result = execute(resumed, tmp_path / "restart", asr=forbidden)
    assert result["outcome"] == "completed"
    assert Counter(o.status for o in outcomes) == {"skipped": 1, "copied": 1}
