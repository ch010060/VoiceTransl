"""Synthetic protocol/crash-consistency cases E01-E12, without inference."""
from __future__ import annotations

import io
import json
from uuid import uuid4

import pytest

from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan
from scripts.asmr_transcribe_lib.gui_protocol import JSONLDecoder, ProtocolError, ReconciliationRequired
from scripts.asmr_transcribe_lib.gui_results import RunStore, atomic_json, read_json, reconcile_run, resume_provenance, safe_ref
from scripts.asmr_transcribe_lib.runtime import CancellationToken, PipelineEvent

SRT = "1\n00:00:00,000 --> 00:00:01,000\nprivate 台詞 never in events\n\n"


def setup_run(tmp_path, count=1, mode="full", asr_metadata=None):
    library = tmp_path / "library"
    library.mkdir()
    for index in range(count):
        audio = library / f"work{index}" / "a.wav"
        audio.parent.mkdir()
        audio.write_bytes(b"synthetic")
    plan = build_execution_plan(library, mode=mode, settings={"parallel": max(1, count)})
    stream = io.StringIO()
    store = RunStore(tmp_path / "run", str(uuid4()), plan, stream, asr_metadata=asr_metadata)
    store.emit("worker_ready", status="ready", payload={"protocol": "1.0", "containment_ready": True})
    store.start()
    return plan, store, stream


def execute(plan, store, fail=False):
    def transcribe(*, audio_path, transcribe_srt_path, **kwargs):
        if fail:
            raise RuntimeError("secret password caption private")
        progress = kwargs.get("transcription_progress_callback")
        if progress:
            progress(1.0, 1.0, 1.0)
        transcribe_srt_path.write_text(SRT, encoding="utf-8")
        return transcribe_srt_path

    def copy(source, target):
        target.write_bytes(source.read_bytes())
        return target

    return run_plan(plan, project_root=plan.root, transcriber=transcribe,
                    translator=copy if plan.mode == "full" else None,
                    opencc_converter=copy, observer=store.observer)


def reconcile(store, exit_code=None):
    return reconcile_run(store.run_dir, store.run_id, store.plan.plan_hash, exit_code)


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_e01_normal_run_receipts_order_and_privacy(tmp_path, mode):
    plan, store, stream = setup_run(tmp_path, mode=mode)
    results = execute(plan, store)
    outcome = store.finish(results)
    assert outcome["outcome"] == "completed"
    assert reconcile(store, 0)["complete"]
    events = JSONLDecoder(store.run_id).feed(stream.getvalue().encode())
    assert events[-1]["event"] == "run_finished"
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert "private" not in stream.getvalue() and str(plan.root) not in stream.getvalue()
    proofs = resume_provenance(store.run_dir, store.run_id, plan.plan_hash)
    assert proofs[0].output_type == ("final" if mode == "full" else "transcription")
    progress = next(i for i, e in enumerate(events) if e["event"] == "asr_progress")
    terminal = next(i for i, e in enumerate(events) if e["event"] == "track_finished")
    assert progress < terminal


def test_e02_interleaved_tracks_have_independent_revisions(tmp_path):
    plan, store, stream = setup_run(tmp_path, count=4)
    store.finish(execute(plan, store))
    decoder = JSONLDecoder(store.run_id)
    events = decoder.feed(stream.getvalue().encode())
    for track in plan.planned_tracks:
        revisions = [e["track_revision"] for e in events if e["track_id"] == track.track_id]
        assert revisions == list(range(1, len(revisions) + 1))
    assert reconcile(store)["counts"]["success"] == 4


def test_e03_split_utf8_and_jsonl(tmp_path):
    _, store, stream = setup_run(tmp_path)
    data = stream.getvalue().encode()
    decoder = JSONLDecoder(store.run_id)
    events = []
    for byte in data:
        events += decoder.feed(bytes([byte]))
    decoder.finish()
    assert len(events) == store.state["last_seq"]
    # A legal unknown minor field can contain split multibyte UTF-8.
    event = json.loads(data.splitlines()[0])
    event["future_field"] = "台詞"
    decoder = JSONLDecoder(store.run_id)
    events = []
    for byte in (json.dumps(event, ensure_ascii=False) + "\n").encode():
        events += decoder.feed(bytes([byte]))
    assert events[0]["future_field"] == "台詞"


def test_e04_identical_duplicate_ignored_conflict_rejected(tmp_path):
    _, store, stream = setup_run(tmp_path)
    first = stream.getvalue().encode().splitlines(keepends=True)[0]
    decoder = JSONLDecoder(store.run_id)
    assert len(decoder.feed(first + first)) == 1
    changed = json.loads(first)
    changed["elapsed_ms"] += 1
    with pytest.raises(ProtocolError, match="conflicting"):
        decoder.feed((json.dumps(changed) + "\n").encode())


def test_e05_sequence_gap_requires_reconciliation(tmp_path):
    _, store, stream = setup_run(tmp_path)
    lines = stream.getvalue().encode().splitlines(keepends=True)
    decoder = JSONLDecoder(store.run_id)
    decoder.feed(lines[0])
    with pytest.raises(ReconciliationRequired):
        decoder.feed(lines[2])
    assert reconcile(store)["outcome"] == "interrupted"


def test_e06_truncated_journal_uses_authoritative_result(tmp_path):
    plan, store, _ = setup_run(tmp_path)
    store.finish(execute(plan, store))
    path = store.run_dir / "events.jsonl"
    path.write_bytes(path.read_bytes()[:-17])
    recovered = reconcile(store, 0)
    assert recovered["complete"] and recovered["counts"]["success"] == 1
    assert "JOURNAL_RECONCILIATION_REQUIRED" in recovered["warnings"]


def test_e06_truncated_frame_and_oversize_rejected(tmp_path):
    _, store, _ = setup_run(tmp_path)
    decoder = JSONLDecoder(store.run_id)
    decoder.feed(b'{"event":')
    with pytest.raises(ProtocolError, match="truncated"):
        decoder.finish()
    with pytest.raises(ProtocolError, match="64 KiB"):
        JSONLDecoder(store.run_id).feed(b"a" * 65537)


def test_e07_exit_zero_does_not_override_failed_track(tmp_path):
    plan, store, stream = setup_run(tmp_path)
    result = store.finish(execute(plan, store, fail=True))
    assert result["exit_code"] == 1
    assert reconcile(store, 0)["outcome"] == "completed_with_errors"
    assert "password" not in stream.getvalue()


def test_e08_crash_without_result_never_success(tmp_path):
    plan, store, _ = setup_run(tmp_path)
    execute(plan, store)
    recovered = reconcile(store, 0)
    assert recovered["counts"]["success"] == 1
    assert not recovered["complete"] and recovered["outcome"] == "interrupted"


def test_e09_missing_journal_notices_reconstruct_from_receipt(tmp_path):
    plan, store, _ = setup_run(tmp_path)
    store.finish(execute(plan, store))
    (store.run_dir / "events.jsonl").write_bytes(b"")
    recovered = reconcile(store, 0)
    assert recovered["complete"] and recovered["counts"]["success"] == 1


def test_e09_receipt_survives_before_state_commit(tmp_path):
    plan, store, _ = setup_run(tmp_path)
    track = plan.planned_tracks[0]
    output = track.task.final_srt_path
    output.write_text(SRT, encoding="utf-8")
    store._terminal_receipt(PipelineEvent("track_finished", track.track_id, status="success"), track)
    recovered = reconcile(store)
    assert recovered["counts"]["success"] == 1
    assert recovered["outcome"] == "interrupted"


@pytest.mark.parametrize("ref", ["../outside.json", "receipts/../../outside", "C:/secret", "/secret", "receipts\\secret", "receipts/./secret"])
def test_e10_reference_traversal_rejected(tmp_path, ref):
    with pytest.raises(ProtocolError):
        safe_ref(tmp_path, ref)


def test_e10_receipt_tamper_rejected(tmp_path):
    plan, store, _ = setup_run(tmp_path)
    store.finish(execute(plan, store))
    receipt_path = next((store.run_dir / "receipts").iterdir())
    receipt = read_json(receipt_path)
    receipt["output_hash"] = "0" * 64
    atomic_json(receipt_path, receipt)
    with pytest.raises(ProtocolError, match="hash"):
        reconcile(store)


def test_e10_output_and_source_changes_rejected(tmp_path):
    plan, store, _ = setup_run(tmp_path)
    store.finish(execute(plan, store))
    plan.planned_tracks[0].task.audio_path.write_bytes(b"changed")
    with pytest.raises(ProtocolError, match="source"):
        reconcile(store)


def test_e11_stale_run_revision_and_unknown_major(tmp_path):
    plan, store, stream = setup_run(tmp_path)
    with pytest.raises(ProtocolError, match="identity"):
        reconcile_run(store.run_dir, str(uuid4()), plan.plan_hash)
    first = json.loads(stream.getvalue().splitlines()[0])
    first["schema_version"] = "2.0"
    with pytest.raises(ProtocolError, match="version"):
        JSONLDecoder(store.run_id).feed((json.dumps(first) + "\n").encode())
    track = plan.planned_tracks[0]
    store.observer(PipelineEvent("track_queued", track.track_id, status="queued"))
    frames = [json.loads(line) for line in stream.getvalue().splitlines()]
    decoder = JSONLDecoder(store.run_id)
    decoder.feed(stream.getvalue().encode())
    last = frames[-1]
    last["seq"] += 1
    with pytest.raises(ProtocolError, match="revision"):
        decoder.feed((json.dumps(last) + "\n").encode())


def test_e12_duplicate_cancel_is_idempotent_and_snapshot_fixed_ref(tmp_path):
    _, store, stream = setup_run(tmp_path)
    token = CancellationToken()
    calls = []
    original = token.cancel
    token.cancel = lambda: (calls.append(True), original())
    command = {"schema_version": "1.0", "run_id": store.run_id, "request_id": str(uuid4()),
               "command": "cancel", "payload": {"reason": "user_stop"}}
    store.ack_control(command, token)
    seq = store.state["last_seq"]
    store.ack_control(command, token)
    store.ack_control({**command, "request_id": str(uuid4())}, token)
    assert calls == [True] and token.cancelled
    assert store.state["last_seq"] == seq + 1
    store.ack_control({**command, "request_id": str(uuid4()), "command": "request_snapshot"}, token)
    assert json.loads(stream.getvalue().splitlines()[-1])["payload"]["snapshot_ref"] == "state.json"
    assert read_json(store.run_dir / "state.json")["status"] == "cancelling"


def test_terminal_receipt_state_and_ledger_precede_stdout(tmp_path):
    plan, store, _ = setup_run(tmp_path)
    class VerifyStream(io.StringIO):
        def write(self, text):
            event = json.loads(text)
            state = read_json(store.run_dir / "state.json")
            assert state["last_seq"] == event["seq"]
            ledger = (store.run_dir / "events.jsonl").read_text(encoding="utf-8")
            assert json.loads(ledger.splitlines()[-1]) == event
            if event["event"] == "track_finished":
                stored = state["tracks"][event["track_id"]]
                assert safe_ref(store.run_dir, stored["receipt_ref"]).exists()
            return super().write(text)
    store.stream = VerifyStream()
    store.finish(execute(plan, store))


def test_files_alone_cannot_create_outcome_and_old_run_not_reused(tmp_path):
    plan, store, _ = setup_run(tmp_path)
    plan.planned_tracks[0].task.final_srt_path.write_text(SRT, encoding="utf-8")
    assert reconcile(store)["counts"]["success"] == 0
    with pytest.raises(FileExistsError):
        RunStore(store.run_dir, store.run_id, plan, io.StringIO())


def test_journal_gap_reconciles_but_unknown_schema_does_not(tmp_path):
    plan, store, _ = setup_run(tmp_path)
    store.finish(execute(plan, store))
    path = store.run_dir / "events.jsonl"
    original = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(original[:2] + original[3:]))
    assert reconcile(store)["complete"]
    event = json.loads(original[0])
    event["schema_version"] = "2.0"
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="schema"):
        reconcile(store)


def test_journal_receipt_traversal_and_result_hash_rejected(tmp_path):
    plan, store, _ = setup_run(tmp_path)
    store.finish(execute(plan, store))
    path = store.run_dir / "events.jsonl"
    events = [json.loads(line) for line in path.read_bytes().splitlines()]
    terminal = next(e for e in events if e["event"] == "track_finished")
    original = terminal["payload"]["receipt_id"]
    terminal["payload"]["receipt_id"] = "../elsewhere.json"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="escapes"):
        reconcile(store)
    terminal["payload"]["receipt_id"] = original
    events[-1]["payload"]["result_hash"] = "0" * 64
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="published result hash"):
        reconcile(store)


def test_receipt_storage_failure_is_fatal_and_never_terminal_notice(tmp_path, monkeypatch):
    import scripts.asmr_transcribe_lib.gui_results as persistence
    from scripts.asmr_transcribe_lib.runtime import ObserverError
    plan, store, stream = setup_run(tmp_path)
    original = persistence.atomic_json
    def fail_receipt(path, payload):
        if path.parent.name == "receipts":
            raise OSError("synthetic storage failure")
        return original(path, payload)
    monkeypatch.setattr(persistence, "atomic_json", fail_receipt)
    with pytest.raises(ObserverError):
        execute(plan, store)
    assert "track_finished" not in stream.getvalue()
    assert not (store.run_dir / "result.json").exists()
    assert reconcile(store)["outcome"] == "interrupted"


def test_lost_stdout_after_receipt_commit_recovers_track(tmp_path):
    from scripts.asmr_transcribe_lib.runtime import ObserverError
    plan, store, _ = setup_run(tmp_path)
    class BrokenTerminal(io.StringIO):
        def write(self, text):
            if json.loads(text)["event"] == "track_finished":
                raise BrokenPipeError("synthetic lost controller")
            return super().write(text)
    store.stream = BrokenTerminal()
    with pytest.raises(ObserverError):
        execute(plan, store)
    recovered = reconcile(store)
    assert recovered["counts"]["success"] == 1
    assert recovered["outcome"] == "interrupted"


@pytest.mark.parametrize("outcome,exit_code", [("preflight_failed", 3), ("plan_rejected", 3), ("protocol_failed", 4), ("internal_error", 4)])
def test_failure_result_has_no_invented_track_terminals(tmp_path, outcome, exit_code):
    _, store, _ = setup_run(tmp_path)
    atomic_json(store.run_dir / "result.json", {
        "schema_version": "1.0", "run_id": store.run_id, "plan_hash": store.plan.plan_hash,
        "outcome": outcome, "exit_code": exit_code, "complete": False,
        "counts": store.snapshot()["counts"], "error": {"code": "SYNTHETIC_FAILURE"}})
    recovered = reconcile(store, exit_code)
    assert recovered["outcome"] == outcome and not recovered["complete"]
    assert len(recovered["missing_track_ids"]) == 1


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_duplicate_output_receipts_preserve_mode_and_dependencies(tmp_path, mode):
    library = tmp_path / "library" / "work"
    library.mkdir(parents=True)
    for suffix in (".wav", ".flac"):
        (library / ("Track01" + suffix)).write_bytes(b"synthetic different variants " + suffix.encode())
    plan = build_execution_plan(library.parent, mode=mode)
    store = RunStore(tmp_path / "run", str(uuid4()), plan, io.StringIO())
    store.start()
    store.finish(execute(plan, store))
    recovered = reconcile(store, 0)
    assert recovered["counts"]["copied"] == 1
    assert recovered["counts"]["success" if mode == "full" else "transcribed"] == 1
    assert all(proof.output_type == ("final" if mode == "full" else "transcription")
               for proof in resume_provenance(store.run_dir, store.run_id, plan.plan_hash))


def test_matching_receipt_hash_cannot_hide_wrong_provenance(tmp_path):
    from scripts.asmr_transcribe_lib.gui_results import _digest
    plan, store, _ = setup_run(tmp_path)
    store.finish(execute(plan, store))
    state = read_json(store.run_dir / "state.json")
    stored = next(iter(state["tracks"].values()))
    receipt_path = safe_ref(store.run_dir, stored["receipt_ref"])
    receipt = read_json(receipt_path)
    receipt["provenance"]["settings_hash"] = "0" * 64
    atomic_json(receipt_path, receipt)
    stored["receipt_hash"] = _digest(receipt)
    atomic_json(store.run_dir / "state.json", state)
    with pytest.raises(ProtocolError, match="provenance"):
        reconcile(store)


def test_snapshot_revision_and_receipt_reference_rejected(tmp_path):
    plan, store, _ = setup_run(tmp_path)
    store.finish(execute(plan, store))
    state = read_json(store.run_dir / "state.json")
    stored = next(iter(state["tracks"].values()))
    stored["revision"] += 1
    atomic_json(store.run_dir / "state.json", state)
    with pytest.raises(ProtocolError, match="revision"):
        reconcile(store)
    stored["revision"] -= 1
    stored["receipt_ref"] = "../outside.json"
    atomic_json(store.run_dir / "state.json", state)
    with pytest.raises(ProtocolError, match="escapes"):
        reconcile(store)


def test_post_completion_output_change_rejected(tmp_path):
    plan, store, _ = setup_run(tmp_path)
    store.finish(execute(plan, store))
    plan.planned_tracks[0].task.final_srt_path.write_text(SRT.replace("private", "changed"), encoding="utf-8")
    with pytest.raises(ProtocolError, match="output hash"):
        reconcile(store)


@pytest.mark.parametrize("code,known,fail,warning", [
    (0, True, False, None),
    (7, True, True, "ASR_NATIVE_NONZERO_EXIT"),
    (None, False, True, "ASR_NATIVE_EXIT_UNKNOWN"),
])
def test_native_asr_exit_metadata_is_durable_and_redacted(tmp_path, code, known, fail, warning):
    metadata = {}
    plan, store, stream = setup_run(tmp_path, asr_metadata=metadata)
    track_id = plan.planned_tracks[0].track_id
    public = {"backend": "xxl", "native_exit_code": code, "returncode_known": known}
    # Mutation after constructing RunStore models worker completion metadata.
    metadata[track_id] = {**public, "stdout": "private caption and key", "stderr": "private native log"}
    store.finish(execute(plan, store, fail=fail))
    recovered = reconcile(store, 1 if fail else 0)
    receipt = read_json(safe_ref(store.run_dir, recovered["tracks"][track_id]["receipt_ref"]))
    assert receipt["native_metadata"] == public
    assert receipt["status"] == ("failed" if fail else "success")
    assert receipt["warnings"] == ([warning] if warning else [])
    # macOS canonical temporary paths include /private/; check actual fixture secrets.
    for secret in ("private caption and key", "private native log", "secret password caption private",
                   "private 台詞 never in events"):
        assert secret not in json.dumps(receipt, ensure_ascii=False) and secret not in stream.getvalue()
    assert recovered["outcome"] == ("completed_with_errors" if fail else "completed")


@pytest.mark.parametrize("metadata", [
    {"backend": "xxl", "native_exit_code": True, "returncode_known": True},
    {"backend": "xxl", "native_exit_code": 7, "returncode_known": False},
    {"backend": "xxl", "native_exit_code": None, "returncode_known": True},
    {"backend": "other", "native_exit_code": 0, "returncode_known": True},
])
def test_reconciliation_rejects_corrupt_native_metadata_even_with_matching_hash(tmp_path, metadata):
    from scripts.asmr_transcribe_lib.gui_results import _digest
    plan, store, _ = setup_run(tmp_path)
    store.finish(execute(plan, store))
    state = read_json(store.run_dir / "state.json")
    stored = next(iter(state["tracks"].values()))
    path = safe_ref(store.run_dir, stored["receipt_ref"])
    receipt = read_json(path)
    receipt["native_metadata"] = metadata
    atomic_json(path, receipt)
    stored["receipt_hash"] = _digest(receipt)
    atomic_json(store.run_dir / "state.json", state)
    with pytest.raises(ProtocolError, match="native ASR"):
        reconcile(store)


@pytest.mark.parametrize("code,known", [(7, True), (None, False)])
def test_successful_terminal_hook_cannot_certify_nonzero_or_unknown_exit(tmp_path, code, known):
    from scripts.asmr_transcribe_lib.runtime import ObserverError
    metadata = {}
    plan, store, stream = setup_run(tmp_path, asr_metadata=metadata)
    metadata[plan.planned_tracks[0].track_id] = {"backend": "xxl", "native_exit_code": code, "returncode_known": known}
    with pytest.raises(ObserverError):
        execute(plan, store)
    assert "track_finished" not in stream.getvalue()
    assert reconcile(store)["outcome"] == "interrupted"
