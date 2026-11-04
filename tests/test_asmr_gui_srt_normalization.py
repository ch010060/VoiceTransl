"""Gate A GUI boundary, authoritative diagnostics and negative-path regression."""
from collections import Counter
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from uuid import uuid4

import pytest

from scripts.asmr_transcribe_lib import gui_execution
from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan, PlanRejected, validate_srt
from scripts.asmr_transcribe_lib.gui_protocol import JSONLDecoder, ProtocolError
from scripts.asmr_transcribe_lib.gui_results import (
    RunStore, read_json, reconcile_run, resume_provenance, safe_ref, atomic_json, _digest,
)
from scripts.asmr_transcribe_lib.local_faster_whisper import FasterWhisperConfig
from scripts.asmr_transcribe_lib.pipeline import process_audio_task
from scripts.asmr_transcribe_lib.opencc_utils import convert_srt_file_opencc
from scripts.asmr_transcribe_lib.runtime import CancellationToken
from scripts.asmr_transcribe_lib.srt_normalization import normalize_srt_bytes, WARNING_CODE
from scripts.asmr_transcribe_lib.srt_utils import format_srt, parse_srt_file, replace_block_text, write_srt_file
from test_asmr_srt_normalization import fixture

RAW = format_srt(fixture()).encode()


def setup(tmp_path, mode="full", count=1, shared=False):
    root = tmp_path / "library"
    for i in range(count):
        directory = root / f"work{i}"
        directory.mkdir(parents=True)
        if shared:
            (directory / "normal").mkdir()
            (directory / "variant").mkdir()
            (directory / "normal/Track01.flac").write_bytes(b"synthetic audio")
            for extension in ("flac", "wav"):
                (directory / "variant" / ("Track01(SEcut)." + extension)).write_bytes(b"synthetic variant")
        else:
            (directory / "Track01.flac").write_bytes(b"synthetic audio")
    plan = build_execution_plan(root, mode=mode, settings={"parallel": 5, "transcribe_parallel": 1})
    assert plan.executable
    stream = io.StringIO()
    store = RunStore(tmp_path / "run", str(uuid4()), plan, stream)
    store.start()
    return plan, store, stream


def execute(plan, store, *, raw=RAW, token=None, mutate=None, archive=None, transcriber=None, calls=None):
    calls = calls if calls is not None else Counter()
    def asr(*, transcribe_srt_path, **kwargs):
        calls["asr"] += 1
        transcribe_srt_path.write_bytes(raw)
        return transcribe_srt_path
    def translate(source, target):
        calls["translation"] += 1
        validate_srt(source)
        blocks = parse_srt_file(source)
        write_srt_file(target, replace_block_text(blocks, ["这是字幕" for _ in blocks]))
        if mutate:
            mutate(source, target)
        return target
    def convert(source, target):
        calls["opencc"] += 1
        return convert_srt_file_opencc(source, target)
    results = run_plan(plan, project_root=plan.root, translator=translate if plan.mode == "full" else None,
                       transcriber=transcriber or asr, opencc_converter=convert,
                       observer=store.observer, cancel_token=token,
                       normalization_store=archive or store.record_normalization)
    result = store.finish(results)
    return results, result


def receipt(store, track):
    return read_json(store.run_dir / "receipts" / (track.track_id + ".json"))


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_rca_end_to_end_receipts_replay_recovery_resume(tmp_path, mode):
    plan, store, stream = setup(tmp_path, mode)
    calls = Counter()
    results, summary = execute(plan, store, calls=calls)
    track, outcome = plan.tracks[0], results[0]
    assert outcome.status == ("success" if mode == "full" else "transcribed")
    assert outcome.warnings == (WARNING_CODE,)
    assert calls == ({"asr": 1, "translation": 1, "opencc": 1} if mode == "full" else {"asr": 1})
    validate_srt(outcome.output_path)
    assert len(parse_srt_file(outcome.output_path)) == 175
    if mode == "full":
        assert "這是字幕" in outcome.output_path.read_text()
    rec = receipt(store, track)
    assert rec["warnings"] == [WARNING_CODE]
    assert rec["output_hash"] == hashlib.sha256(outcome.output_path.read_bytes()).hexdigest()
    audit = read_json(safe_ref(store.run_dir, rec["timestamp_normalization"]["ref"]))
    assert audit["raw_hash"] == hashlib.sha256(RAW).hexdigest()
    assert audit["normalized_hash"] == hashlib.sha256(normalize_srt_bytes(RAW)[0]).hexdigest()
    assert safe_ref(store.run_dir, audit["raw_ref"]).read_bytes() == RAW
    assert audit["repaired_cues"] == 1 and audit["max_displacement_ms"] == 420
    assert audit["changes"][0]["normalized_timestamp"] == "00:12:05,460 --> 00:12:19,000"
    assert summary["complete"] and reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]
    proofs = resume_provenance(store.run_dir, store.run_id, plan.plan_hash)
    resumed = build_execution_plan(plan.root, mode=mode, provenance=proofs)
    assert resumed.executable
    assert not list(plan.root.rglob("*.normalized.srt")) and not list(plan.root.rglob("*.tmp"))
    events = JSONLDecoder(store.run_id).feed(stream.getvalue().encode())
    terminal = next(e for e in events if e["event"] == "track_finished")
    assert terminal["payload"]["warnings"] == [WARNING_CODE]
    assert "Synthetic" not in stream.getvalue() and str(plan.root) not in stream.getvalue()
    # Replay is read-only and does not normalize/write an already committed result.
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]


def test_valid_asr_is_byte_stable_and_has_no_archive(tmp_path):
    plan, store, _ = setup(tmp_path, "transcribe_only")
    raw = b"\xef\xbb\xbf1\r\n00:00:00,000 --> 00:00:00,000\r\ntext\r\n"
    results, _ = execute(plan, store, raw=raw)
    assert results[0].output_path.read_bytes() == raw
    assert not results[0].warnings and not store.normalizations
    assert "timestamp_normalization" not in receipt(store, plan.tracks[0])


@pytest.mark.parametrize("raw,category", [
    (RAW.replace(b"00:12:05,040", b"00:12:04,459"), "displacement_limit"),
    (b"WEBVTT", "format"), (b"", "empty"),
    (RAW.replace(b"00:12:19,000", b"00:12:05,450"), "repair_exceeds_end"),
])
def test_policy_failure_codes_and_no_translation(tmp_path, raw, category):
    plan, store, stream = setup(tmp_path)
    calls = Counter()
    results, _ = execute(plan, store, raw=raw, calls=calls)
    assert results[0].status == "failed" and calls == {"asr": 1}
    assert plan.tracks[0].task.transcribe_srt_path.read_bytes() == raw
    rec = receipt(store, plan.tracks[0])
    assert rec["error_code"] == "SRT_TIMESTAMP_ORDER_INVALID" and rec["timestamp_error"] == category
    assert rec["provenance"] is None and not rec["warnings"]
    error = next(e for e in JSONLDecoder(store.run_id).feed(stream.getvalue().encode()) if e["event"] == "track_failed")
    assert error["payload"]["error"]["code"] == "SRT_TIMESTAMP_ORDER_INVALID"


@pytest.mark.parametrize("code", [0, 7, None])
def test_native_exit_evidence_gates_normalization(tmp_path, monkeypatch, code):
    from scripts.asmr_transcribe_lib import local_faster_whisper as native
    from scripts.asmr_transcribe_lib.gui_native_adapter import strict_xxl_transcriber
    plan, store, _ = setup(tmp_path)
    def native_run(**kwargs):
        kwargs["subprocess_run"]([], cwd=plan.root)
        kwargs["transcribe_srt_path"].write_bytes(RAW)
        return kwargs["transcribe_srt_path"]
    monkeypatch.setattr(native, "_run_with_progress", lambda *args: (code, "private log"))
    monkeypatch.setattr(native, "run_faster_whisper_transcription", native_run)
    results, _ = execute(plan, store, transcriber=strict_xxl_transcriber(plan, store.asr_metadata))
    rec = receipt(store, plan.tracks[0])
    if code == 0:
        assert results[0].status == "success" and store.normalizations
        assert rec["provenance"] and rec["warnings"] == [WARNING_CODE]
    else:
        assert results[0].status == "failed" and not store.normalizations
        assert rec["provenance"] is None and rec["warnings"] == ["ASR_NATIVE_EXIT_UNKNOWN" if code is None else "ASR_NATIVE_NONZERO_EXIT"]


@pytest.mark.parametrize("fault", ["cancel", "archive_io", "publish_io", "output_drift", "source_drift", "temp_collision"])
def test_prepublication_failure_preserves_raw_without_certification(tmp_path, monkeypatch, fault):
    plan, store, _ = setup(tmp_path)
    track = plan.tracks[0]
    token = CancellationToken()
    original_replace = gui_execution.os.replace
    def replace_file(source, target):
        if fault == "publish_io" and Path(target) == track.task.transcribe_srt_path:
            raise OSError("synthetic disk failure")
        return original_replace(source, target)
    monkeypatch.setattr(gui_execution.os, "replace", replace_file)
    def archive(*args):
        if fault == "archive_io":
            raise OSError("synthetic evidence disk failure")
        store.record_normalization(*args)
        if fault == "cancel":
            token.cancel()
        elif fault == "output_drift":
            track.task.transcribe_srt_path.write_bytes(b"foreign changed output")
        elif fault == "source_drift":
            track.task.audio_path.write_bytes(b"foreign changed source")
        elif fault == "temp_collision":
            track.task.transcribe_srt_path.with_suffix(".srt.tmp").write_bytes(b"foreign temporary")
    results, _ = execute(plan, store, token=token, archive=archive)
    assert results[0].status == ("cancelled" if fault == "cancel" else "failed")
    rec = receipt(store, track)
    assert rec["provenance"] is None and rec["output_hash"] is None
    assert not track.task.final_srt_path.exists()
    assert track.task.transcribe_srt_path.read_bytes() == (b"foreign changed output" if fault == "output_drift" else RAW)
    if fault == "temp_collision":
        assert track.task.transcribe_srt_path.with_suffix(".srt.tmp").read_bytes() == b"foreign temporary"


@pytest.mark.parametrize("fault", ["source", "transcript"])
def test_drift_after_normalization_fails_closed(tmp_path, fault):
    plan, store, _ = setup(tmp_path)
    def mutate(source, target):
        if fault == "source":
            plan.tracks[0].task.audio_path.write_bytes(b"changed source")
        else:
            source.write_bytes(source.read_bytes().replace(b"Synthetic", b"Different"))
    results, _ = execute(plan, store, mutate=mutate)
    assert results[0].status == "failed"
    assert receipt(store, plan.tracks[0])["provenance"] is None
    assert not plan.tracks[0].task.final_srt_path.exists()


@pytest.mark.parametrize("target", ["raw_ref", "normalized_ref", "audit", "escape", "missing_warning", "missing_audit"])
def test_recovery_refuses_tampered_normalization_evidence(tmp_path, target):
    plan, store, _ = setup(tmp_path, "transcribe_only")
    execute(plan, store)
    rec = receipt(store, plan.tracks[0])
    audit_path = safe_ref(store.run_dir, rec["timestamp_normalization"]["ref"])
    audit = read_json(audit_path)
    if target in {"raw_ref", "normalized_ref"}:
        safe_ref(store.run_dir, audit[target]).write_bytes(b"tampered")
    elif target == "missing_audit":
        audit_path.unlink()
    elif target == "audit":
        audit["repaired_cues"] = 2
        atomic_json(audit_path, audit)
    else:
        if target == "escape":
            rec["timestamp_normalization"]["ref"] = "../outside.json"
        else:
            rec["warnings"] = []
        atomic_json(store.run_dir / "receipts" / (plan.tracks[0].track_id + ".json"), rec)
        store.state["tracks"][plan.tracks[0].track_id]["receipt_hash"] = _digest(rec)
        store._save_state()
    with pytest.raises(ProtocolError):
        reconcile_run(store.run_dir, store.run_id, plan.plan_hash)


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_shared_owner_parallel_five_asr_one(tmp_path, mode):
    plan, store, _ = setup(tmp_path, mode, count=5, shared=True)
    calls = Counter()
    results, summary = execute(plan, store, calls=calls)
    assert calls["asr"] == 5 and len(store.normalizations) == 5
    assert summary["counts"]["copied"] == 10
    assert sum(bool(t.shared_output_owner) for t in plan.tracks) == 5
    for track in plan.tracks:
        rec = receipt(store, track)
        if track.shared_output_owner:
            assert rec["status"] == "copied" and "shared_output_receipt" in rec
            assert "timestamp_normalization" not in rec
        elif track.task.action == "process":
            assert rec["warnings"] == [WARNING_CODE]
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]
    assert all(r.status in {"success", "transcribed", "copied"} for r in results)


def test_orphan_invalid_old_transcript_is_not_adopted_or_overwritten(tmp_path):
    plan, store, _ = setup(tmp_path, "transcribe_only")
    transcript = plan.tracks[0].task.transcribe_srt_path
    transcript.write_bytes(RAW)
    with pytest.raises(PlanRejected):
        execute(plan, store)
    replanned = build_execution_plan(plan.root, mode="transcribe_only")
    assert any(i.code == "UNVERIFIED_LEGACY" for i in replanned.issues)
    assert transcript.read_bytes() == RAW


def test_cli_default_does_not_normalize(tmp_path):
    plan, _, _ = setup(tmp_path, "transcribe_only")
    def asr(**kwargs):
        kwargs["transcribe_srt_path"].write_bytes(RAW)
        return kwargs["transcribe_srt_path"]
    # Legacy CLI has no strict GUI validator; preserve its exact historical bytes/policy.
    result = process_audio_task(plan.tracks[0].task, project_root=plan.root,
                                faster_whisper_config=FasterWhisperConfig(model="synthetic", language="ja"), transcriber=asr)
    assert result.output_path.read_bytes() == RAW
    with pytest.raises(ValueError, match="timestamps out of order"):
        validate_srt(result.output_path)


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_resume_never_normalizes_or_changes_committed_output(tmp_path, mode):
    plan, store, _ = setup(tmp_path, mode)
    outcomes, _ = execute(plan, store)
    output = outcomes[0].output_path
    stamp = output.stat()
    original = output.read_bytes()
    proofs = resume_provenance(store.run_dir, store.run_id, plan.plan_hash)
    resumed = build_execution_plan(plan.root, mode=mode, provenance=proofs)
    next_store = RunStore(tmp_path / "resumed-run", str(uuid4()), resumed, io.StringIO())
    next_store.start()
    calls = Counter()
    results, summary = execute(resumed, next_store, calls=calls)
    assert results[0].status in {"skipped", "transcribed"} and summary["complete"]
    assert calls["asr"] == 0 and not next_store.normalizations
    assert output.read_bytes() == original and output.stat().st_mtime_ns == stamp.st_mtime_ns
    assert receipt(store, plan.tracks[0])["warnings"] == [WARNING_CODE]


def test_text_correction_follows_normalization_and_raw_is_preserved(tmp_path):
    plan, store, _ = setup(tmp_path, "transcribe_only")
    config = plan.root / "config"
    config.mkdir()
    (config / "asmr_transcript_corrections.json").write_text(json.dumps({"rules": [{"wrong": "Synthetic", "correct": "Corrected"}]}))
    results, _ = execute(plan, store)
    assert results[0].status == "transcribed"
    assert b"Corrected" in results[0].output_path.read_bytes()
    rec = receipt(store, plan.tracks[0])
    audit = read_json(safe_ref(store.run_dir, rec["timestamp_normalization"]["ref"]))
    assert safe_ref(store.run_dir, audit["raw_ref"]).read_bytes() == RAW
    assert b"Synthetic" in safe_ref(store.run_dir, audit["normalized_ref"]).read_bytes()
    assert audit["normalized_hash"] != rec["output_hash"]  # Text correction is a later stage.
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]


def test_no_trusted_store_refuses_repair(tmp_path):
    plan, store, _ = setup(tmp_path, "transcribe_only")
    def asr(**kwargs):
        kwargs["transcribe_srt_path"].write_bytes(RAW)
        return kwargs["transcribe_srt_path"]
    results = run_plan(plan, project_root=plan.root, translator=None, transcriber=asr,
                       opencc_converter=lambda *_: None, observer=store.observer)
    store.finish(results)
    assert results[0].status == "failed"
    assert receipt(store, plan.tracks[0])["timestamp_error"] == "audit_store_required"
    assert plan.tracks[0].task.transcribe_srt_path.read_bytes() == RAW


def test_archive_namespace_symlink_cannot_escape_run(tmp_path):
    plan, store, _ = setup(tmp_path, "transcribe_only")
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.run_dir / "logs/srt-normalization").symlink_to(outside, target_is_directory=True)
    results, _ = execute(plan, store)
    assert results[0].status == "failed" and not list(outside.iterdir())
    assert plan.tracks[0].task.transcribe_srt_path.read_bytes() == RAW


def test_evidence_is_not_rewritten_after_terminal_receipt(tmp_path):
    plan, store, _ = setup(tmp_path, "transcribe_only")
    execute(plan, store)
    before = {p: p.read_bytes() for p in store.run_dir.rglob("*") if p.is_file()}
    normalized, changes = normalize_srt_bytes(RAW)
    with pytest.raises(ProtocolError):
        store.record_normalization(plan.tracks[0].track_id, RAW, normalized, changes)
    assert {p: p.read_bytes() for p in before} == before


def test_cancel_after_atomic_repair_retains_evidence_without_success_receipt(tmp_path):
    plan, store, _ = setup(tmp_path, "transcribe_only")
    token = CancellationToken()
    observer = store.observer
    def cancel(event):
        observer(event)
        if event.event == "asr_completed":
            token.cancel()
    store.observer = cancel
    results, _ = execute(plan, store, token=token)
    assert results[0].status == "cancelled"
    assert receipt(store, plan.tracks[0])["provenance"] is None
    assert receipt(store, plan.tracks[0])["warnings"] == [WARNING_CODE]
    assert plan.tracks[0].task.transcribe_srt_path.read_bytes() == normalize_srt_bytes(RAW)[0]


def test_output_drift_between_asr_completion_and_correction_is_rejected(tmp_path):
    plan, store, _ = setup(tmp_path, "transcribe_only")
    observer = store.observer
    def mutate(event):
        observer(event)
        if event.event == "asr_completed":
            plan.tracks[0].task.transcribe_srt_path.write_bytes(normalize_srt_bytes(RAW)[0].replace(b"Synthetic", b"Foreign"))
    store.observer = mutate
    results, _ = execute(plan, store)
    assert results[0].status == "failed"
    assert receipt(store, plan.tracks[0])["provenance"] is None
