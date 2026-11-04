"""Gate A/B regression. Real RCA timings, synthetic inference, explicit barriers."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import io
from pathlib import Path
import shutil
import threading
from uuid import uuid4

import pytest

from scripts.asmr_transcribe_lib import gui_execution, pipeline
from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan, validate_srt
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


@pytest.mark.parametrize("boundary,code,stage", [
    ("asr", "ASR_EXECUTION_FAILED", "asr"),
    ("translation", "TRANSLATION_FAILED", "translation"),
    ("opencc", "OPENCC_FAILED", "opencc"),
    ("publication", "OUTPUT_PUBLICATION_FAILED", "finalize"),
])
def test_failure_categories_have_exact_track_context_without_private_exception(tmp_path, monkeypatch, boundary, code, stage):
    plan = build_execution_plan(library(tmp_path))
    def fail(*args, **kwargs):
        raise RuntimeError("PRIVATE_CAPTION_DO_NOT_LOG token=PRIVATE_SECRET https://private.invalid")
    kwargs = {"asr": fail} if boundary == "asr" else {"translate": fail} if boundary == "translation" else {"convert": fail} if boundary == "opencc" else {}
    if boundary == "publication":
        monkeypatch.setattr(gui_execution, "_publish_new", fail)
    store, outcomes, result = execute(plan, tmp_path / "run", **kwargs)
    failed = next(o for o in outcomes if o.status == "failed")
    receipt = read_json(store.run_dir / "receipts" / (failed.track_id + ".json"))
    error = receipt["error"]
    assert error["code"] == code and error["stage"] == stage
    assert error["track_id"] == failed.track_id and error["work_id"] == plan.track_index[failed.track_id].work_id
    assert error["audio_path"] == failed.audio_path.name and error["outcome"] == "failed"
    assert (store.run_dir / error["diagnostic_ref"]).exists()
    assert result["counts"]["not_started"] == 1
    dependency = next(o for o in outcomes if o.status == "not_started")
    assert read_json(store.run_dir / "receipts" / (dependency.track_id + ".json"))["error_code"] == "DEPENDENCY_FAILED"
    text = "\n".join(p.read_text(encoding="utf-8") for p in store.run_dir.rglob("*") if p.is_file() and p.name != ".store")
    assert "PRIVATE_CAPTION" not in text and "PRIVATE_SECRET" not in text and "private.invalid" not in text
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]


def test_missing_asr_output_is_distinct(tmp_path):
    plan = build_execution_plan(library(tmp_path, variants=False))
    store, outcomes, _ = execute(plan, tmp_path / "run", asr=lambda **kw: kw["transcribe_srt_path"])
    assert outcomes[0].failure["code"] == "ASR_OUTPUT_MISSING"


@pytest.mark.parametrize("timings,valid", [
    (["00:12:05,460 --> 00:12:05,560", "00:12:05,040 --> 00:12:19,000"], False),
    (["00:00:30,000 --> 00:00:31,000", "00:00:00,000 --> 00:00:01,000"], False),
    (["00:00:03,000 --> 00:00:04,000", "00:00:02,500 --> 00:00:04,000", "00:00:02,000 --> 00:00:04,000"], False),
    (["00:00:01,000 --> 00:00:03,000", "00:00:02,000 --> 00:00:04,000"], True),
    (["00:00:01,000 --> 00:00:01,000"], True),
    (["00:00:02,000 --> 00:00:01,000"], False),
])
def test_timestamp_repair_deferred_original_text_and_order_preserved(tmp_path, timings, valid):
    plan = build_execution_plan(library(tmp_path, variants=False), mode="transcribe_only")
    original = "\n\n".join(f"{i}\n{timing}\nPRIVATE_CAPTION_{i}" for i, timing in enumerate(timings, 1)) + "\n"
    def asr(*, transcribe_srt_path, **kwargs):
        transcribe_srt_path.write_text(original, encoding="utf-8")
        return transcribe_srt_path
    store, outcomes, _ = execute(plan, tmp_path / "run", asr=asr)
    transcript = plan.tracks[0].task.transcribe_srt_path
    assert transcript.read_text(encoding="utf-8") == original
    assert outcomes[0].status == ("transcribed" if valid else "failed")
    if not valid:
        error = read_json(store.run_dir / "receipts" / (plan.tracks[0].track_id + ".json"))["error"]
        assert error["code"] == "SRT_TIMESTAMP_ORDER_INVALID"
        assert error["details"]["automatic_repair"] == "DEFERRED"
        if len(timings) == 3:
            assert error["details"]["anomaly_count"] == 2
        if "12:05" in timings[0]:
            assert error["details"]["examples"][0]["backward_ms"] == 420
    else:
        validate_srt(transcript)  # Preserve existing overlap/zero-length policy.


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


def test_aborted_not_started_records_are_not_user_cancelled(tmp_path):
    plan = build_execution_plan(library(tmp_path))
    token = CancellationToken()
    token.cancel("run_aborted")
    store, outcomes, result = execute(plan, tmp_path / "run", token=token)
    assert result["outcome"] == "interrupted"
    assert all(o.status == "not_started" and o.reason == "run_aborted" for o in outcomes)
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["outcome"] == "interrupted"


def test_readable_log_failure_does_not_fail_a_track(tmp_path, monkeypatch):
    from scripts.asmr_transcribe_lib.gui_logging import DiagnosticLog
    monkeypatch.setattr(DiagnosticLog, "append", lambda *args: (_ for _ in ()).throw(OSError("PRIVATE_SECRET")))
    plan = build_execution_plan(library(tmp_path))
    _, _, result = execute(plan, tmp_path / "run")
    assert result["outcome"] == "completed"


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


def test_skip_only_work_has_one_truthful_summary(tmp_path):
    root = library(tmp_path)
    (root / "RJ001/voice01.srt").write_text(SRT, encoding="utf-8")
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    def forbidden(**kwargs):
        raise AssertionError("skip-only Work must not invoke ASR")
    store, _, result = execute(plan, tmp_path / "run", asr=forbidden)
    assert result["counts"]["skipped"] == 2
    log = (store.run_dir / "logs/execution.log").read_text(encoding="utf-8")
    assert log.count("作品結算") == 1 and "略過 2" in log
