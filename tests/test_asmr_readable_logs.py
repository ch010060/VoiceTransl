"""Gate C: frozen identity, authoritative terminals and bounded Qt presentation."""
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtCore import QObject, QSettings, QTimer, pyqtSignal

from scripts.asmr_gui.controller import Controller
from scripts.asmr_gui.window import BatchWindow
from scripts.asmr_transcribe_lib.gui_logging import RunLogFormatter
from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan
from scripts.asmr_transcribe_lib.gui_results import read_json, reconcile_run, atomic_json
from scripts.asmr_transcribe_lib.gui_protocol import ProtocolError
from test_asmr_runtime_closure import library, execute


class QuietHealth(QObject):
    updated = pyqtSignal(dict)
    def __init__(self, **kwargs):
        super().__init__(kwargs.get("parent"))
    def configure(self, *args):
        pass
    close = configure
    set_paused = configure


@pytest.fixture
def view(tmp_path, qtbot, monkeypatch):
    import scripts.asmr_gui.window as module
    monkeypatch.setattr(module, "QSettings", lambda *args: QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))
    controller = Controller(state_root=tmp_path / "state", health_monitor_factory=QuietHealth)
    window = BatchWindow(controller)
    qtbot.addWidget(window)
    yield controller, window
    if controller.busy:
        controller._set_state("ERROR")
    assert controller.request_close()
    controller._diagnostics.close(wait=True)
    controller._diagnostics.thread.join(timeout=10)
    assert not controller._diagnostics.thread.is_alive()


def event(formatter, track, name, revision, *, status="running", stage=None, payload=None, elapsed=1000):
    return {"run_id": formatter.run_id, "work_id": track.work_id, "track_id": track.track_id,
            "emitted_at": "2026-11-03T15:40:00.118Z", "event": name,
            "elapsed_ms": elapsed, "track_revision": revision,
            "status": status, "stage": stage, "payload": payload or {}}


def test_gui_and_full_log_have_work_track_stage_error_and_receipt(view, qtbot, tmp_path):
    controller, window = view
    root = library(tmp_path)
    plan = build_execution_plan(root)
    controller.set_root(str(root))
    controller.run_model.load_plan(plan)
    intent = controller._new_intent("run")
    formatter = RunLogFormatter(plan, str(uuid4()))
    track = next(t for t in plan.tracks if t.task.action == "process")
    other = next(t for t in plan.tracks if t.track_id != track.track_id)
    lines = []
    lines += formatter.event(event(formatter, track, "track_started", 1))
    lines += formatter.event(event(formatter, track, "stage_changed", 2, stage="asr"))
    for revision in range(3, 103):
        assert not formatter.event(event(formatter, track, "stage_changed", revision, stage="asr"))
    lines += formatter.event(event(formatter, track, "track_failed", 103, status="failed", payload={
        "receipt_id": "receipts/" + track.track_id + ".json",
        "error": {"code": "SRT_TIMESTAMP_ORDER_INVALID", "detail_ref": "logs/" + track.track_id + ".error.json"}}))
    lines += formatter.event(event(formatter, other, "track_not_started", 1, status="not_started", payload={
        "reason": "dependency_failed", "receipt_id": "receipts/" + other.track_id + ".json"}))
    # Filtering changes the tree only; formatting always uses frozen Track IDs.
    window.search_edit.setText("a search with no matching rows")
    controller._post(intent, "run_log", lines)
    qtbot.waitUntil(lambda: "SRT_TIMESTAMP_ORDER_INVALID" in window.log_view.toPlainText())
    text = window.log_view.toPlainText()
    assert "RJ001" in text and track.task.audio_path.name in text
    assert "開始語音辨識" in text and "依賴失敗" in text
    assert text.count("作品結算") == 1 and "作品結算（含失敗）" in text
    assert "失敗 1" in text and "未執行 1" in text
    assert "waiting_asr" not in text and "stage_changed" not in text
    controller._diagnostics.close(wait=True)
    full = controller.diagnostic_log_path.read_text(encoding="utf-8")
    assert all(value in full for value in [formatter.run_id, track.work_id, track.track_id, track.task.audio_path.name,
                                          "stage=asr", "SRT_TIMESTAMP_ORDER_INVALID", "receipts/", "diagnostic=logs/"])


def test_receipt_required_for_work_completion_and_no_repeated_summary(tmp_path):
    plan = build_execution_plan(library(tmp_path, variants=False))
    formatter = RunLogFormatter(plan, str(uuid4()))
    track = plan.tracks[0]
    assert not formatter.event(event(formatter, track, "track_finished", 1, status="success"))
    proper = event(formatter, track, "track_finished", 2, status="success", payload={"receipt_id": "receipts/" + track.track_id + ".json"})
    assert sum("作品結算" in message for message, _ in formatter.event(proper)) == 1
    assert not formatter.event(proper)
    assert not formatter.event({**proper, "track_revision": 3})


def test_worker_readable_log_has_all_outcomes_and_does_not_replay_per_update(tmp_path, monkeypatch):
    plan = build_execution_plan(library(tmp_path, works=("RJ001", "RJ002")))
    store, outcomes, result = execute(plan, tmp_path / "run")
    full = (store.run_dir / "logs/execution.log").read_text(encoding="utf-8")
    assert "作品結算" in full and "完成翻譯" in full and "複製完成" in full
    assert "owner=" in full and "copy_source=" in full and "receipt=" in full
    assert "PRIVATE_CAPTION" not in full
    assert all(track.track_id in full for track in plan.tracks)
    original = Path.open
    def no_journal(path, *args, **kwargs):
        assert path.name != "events.jsonl", "live receipt recovery replayed the journal"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_journal)
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash, verify_journal=False)["complete"]
    # Live projection still rejects altered authoritative receipts.
    receipt_path = store.run_dir / "receipts" / (plan.tracks[0].track_id + ".json")
    receipt = read_json(receipt_path)
    receipt["status"] = "failed"
    atomic_json(receipt_path, receipt)
    with pytest.raises(ProtocolError):
        reconcile_run(store.run_dir, store.run_id, plan.plan_hash, verify_journal=False)


@pytest.mark.parametrize("outcome,complete,stop,expected", [
    ("completed", True, False, "執行完成"),
    ("completed_with_errors", True, False, "含音軌失敗"),
    ("interrupted", False, False, "Worker 異常退出"),
    ("interrupted", True, False, "Worker 異常退出"),
    ("interrupted", False, True, "使用者要求停止"),
    ("protocol_failed", False, False, "通訊協定失敗"),
])
def test_batch_summary_distinguishes_causes_and_cleanup(tmp_path, outcome, complete, stop, expected):
    plan = build_execution_plan(library(tmp_path))
    formatter = RunLogFormatter(plan, str(uuid4()))
    result = {"run_id": formatter.run_id, "plan_hash": plan.plan_hash, "outcome": outcome,
              "complete": complete, "tracks": {}, "counts": {"not_started": 2}}
    lines = formatter.reconcile(result, {"cleanup_verified": True, "active_count": 0}, user_stop=stop)
    assert expected in lines[-1][0]
    assert "cleanup=True active=0" in lines[-1][0] and "未執行 2" in lines[-1][0]
    if not stop and outcome == "interrupted":
        assert "使用者要求停止" not in lines[-1][0]


def test_large_log_remains_bounded_and_qt_timer_runs(view, qtbot, tmp_path):
    controller, window = view
    root = library(tmp_path, variants=False)
    base = build_execution_plan(root)
    # 1,151 distinct stable identities; no model work or disk discovery required.
    original = base.tracks[0]
    tracks = tuple(replace(original, track_id=str(uuid4()), ordinal=i,
                           task=replace(original.task, audio_path=original.task.audio_path.with_name(f"voice{i:04}.flac")))
                   for i in range(1151))
    work = replace(base.works[0], track_ids=tuple(t.track_id for t in tracks))
    plan = replace(base, tracks=tracks, works=(work,))
    formatter = RunLogFormatter(plan, str(uuid4()))
    controller.set_root(str(root))
    controller.run_model.load_plan(plan)
    intent = controller._new_intent("run")
    tick = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: tick.append("作品結算" not in window.log_view.toPlainText()))
    timer.start()
    for track in tracks:
        lines = formatter.event(event(formatter, track, "track_started", 1))
        lines += formatter.event(event(formatter, track, "track_finished", 2, status="success",
                                        payload={"receipt_id": "receipts/" + track.track_id + ".json"}))
        controller._post(intent, "run_log", lines)
    qtbot.waitUntil(lambda: "作品結算" in window.log_view.toPlainText(), timeout=10000)
    qtbot.waitUntil(lambda: bool(tick))
    timer.stop()
    assert window.log_view.document().blockCount() <= 1500
    assert len(formatter.terminal) == 1151
    assert any(tick), "Qt timers did not run while logs were being delivered"


def test_invalid_protocol_text_is_not_displayed(tmp_path):
    plan = build_execution_plan(library(tmp_path, variants=False))
    formatter = RunLogFormatter(plan, str(uuid4()))
    track = plan.tracks[0]
    lines = formatter.event(event(formatter, track, "track_failed", 1, status="failed", payload={
        "receipt_id": "receipts/" + track.track_id + ".json",
        "error": {"code": "PRIVATE_CAPTION token=PRIVATE_SECRET", "detail_ref": "https://secret.invalid"}}))
    text = str(lines)
    assert "PRIVATE_CAPTION" not in text and "PRIVATE_SECRET" not in text and "secret.invalid" not in text


def test_repaired_timestamp_success_has_truthful_readable_warning(tmp_path):
    from test_asmr_gui_srt_normalization import RAW
    from scripts.asmr_transcribe_lib.srt_normalization import WARNING_CODE
    plan = build_execution_plan(library(tmp_path, variants=False))
    def asr(*, transcribe_srt_path, **kwargs):
        transcribe_srt_path.write_bytes(RAW)
        return transcribe_srt_path
    store, outcomes, result = execute(plan, tmp_path / "run", asr=asr)
    assert result["outcome"] == "completed" and outcomes[0].warnings == (WARNING_CODE,)
    text = (store.run_dir / "logs/execution.log").read_text(encoding="utf-8")
    assert WARNING_CODE in text and "完成翻譯" in text and "DEFERRED" not in text
    assert "Synthetic" not in text


def test_reconciled_receipt_retains_normalization_warning(view, tmp_path):
    from test_asmr_gui_srt_normalization import RAW
    controller, _ = view
    plan = build_execution_plan(library(tmp_path, variants=False))
    def asr(*, transcribe_srt_path, **kwargs):
        transcribe_srt_path.write_bytes(RAW)
        return transcribe_srt_path
    store, _, _ = execute(plan, tmp_path / "run", asr=asr)
    recovered = controller._enrich(reconcile_run(store.run_dir, store.run_id, plan.plan_hash), store.run_dir)
    formatter = RunLogFormatter(plan, store.run_id)
    lines = formatter.reconcile(recovered, {"cleanup_verified": True, "active_count": 0})
    assert "SRT_TIMESTAMP_NORMALIZED" in str(lines) and "DEFERRED" not in str(lines)
