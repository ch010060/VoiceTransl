"""Phase 1C synthetic GUI matrix. No ASR model or Sakura inference is used."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QObject, QPoint, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication, QProgressBar, QWidget

from scripts.asmr_gui.run_model import RunModel
from scripts.asmr_gui.tree_model import ROLE_NODE, WorkFilterProxyModel, WorkTrackModel
from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan

PROJECT = Path(__file__).resolve().parents[1]
DRIVER = PROJECT / "tests" / "fixtures" / "asmr_gui_v1_driver.py"


class SilentHealthMonitor(QObject):
    """Keep baseline lifecycle tests independent of machine backend availability."""
    updated = pyqtSignal(dict)

    def __init__(self, *, project_root, parent=None):
        super().__init__(parent)

    def configure(self, settings, mode="full"):
        pass

    def check(self):
        return True

    def set_paused(self, paused):
        pass

    def close(self):
        pass


@pytest.fixture(scope="module")
def app():
    from scripts.asmr_gui.theme import apply_theme
    instance = QApplication.instance() or QApplication([])
    instance.setQuitOnLastWindowClosed(False)
    apply_theme(instance)
    return instance


@pytest.fixture(autouse=True)
def isolated_gui_preferences(tmp_path, monkeypatch):
    """Prevent synthetic GUI tests from changing the user's application prefs."""
    import scripts.asmr_gui.window as window_module
    import scripts.asmr_gui.controller as controller_module
    monkeypatch.setattr(controller_module, "HealthMonitor", SilentHealthMonitor, raising=False)
    monkeypatch.setattr(window_module, "QSettings", lambda *args: QSettings(
        str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat))


def wait_for(app, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.005)
    app.processEvents()
    assert predicate(), "bounded GUI condition timed out"


def library(tmp_path, names=("RJ001/voice01.wav", "RJ001/nested/voice02.flac", "RJ002/voice03.wav")):
    root = tmp_path / "library"
    root.mkdir()
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SYNTHETIC_NOT_REAL_AUDIO")
    return root


@pytest.fixture
def catalog(tmp_path):
    root = library(tmp_path)
    (root / "empty").mkdir()
    return build_execution_plan(root)


def model_for(plan):
    run_model = RunModel()
    run_model.load_plan(plan)
    return run_model, WorkTrackModel(run_model)


def track_index(tree, track_id):
    for row in range(tree.rowCount()):
        parent = tree.index(row, 0)
        for child in range(tree.rowCount(parent)):
            index = tree.index(child, 0, parent)
            if index.data(ROLE_NODE)["id"] == track_id:
                return index
    raise AssertionError("Track identity missing from tree")


def event(model, track, name, *, revision=1, stage=None, status="running", elapsed=100, payload=None, run_id=None):
    model.apply_event({"schema_version": "1.0", "event": name,
        "run_id": run_id or model.run_id, "seq": revision + 1,
        "emitted_at": "2026-11-02T00:00:00.000Z", "elapsed_ms": elapsed,
        "work_id": track.work_id, "track_id": track.track_id,
        "track_revision": revision, "stage": stage, "status": status,
        "payload": payload or {}})


def test_v01_v02_scan_hierarchy_relative_paths(app, catalog):
    model, tree = model_for(catalog)
    assert tree.columnCount() == 4
    assert tree.rowCount() == 3
    assert sum(tree.rowCount(tree.index(i, 0)) for i in range(tree.rowCount())) == 3
    nested = next(t for t in catalog.tracks if "nested" in str(t.task.audio_path))
    index = track_index(tree, nested.track_id)
    assert "nested" in index.data() or "nested" in index.data(Qt.ItemDataRole.ToolTipRole)
    assert model.summary()["total_works"] == 3


def test_v03_synthetic_root_and_real_root_are_distinct(app, tmp_path):
    plan = build_execution_plan(library(tmp_path, ("root.wav", "_root/voice.wav")))
    model, tree = model_for(plan)
    assert len(plan.works) == 2
    assert len({w.work_id for w in plan.works}) == 2
    assert {w.kind for w in plan.works} == {"synthetic_root", "directory"}
    assert {tree.index(i, 0).data(ROLE_NODE)["id"] for i in range(2)} == {w.work_id for w in plan.works}
    assert model.summary()["planned"] == 2


def test_v04_v06_work_selection_only_and_empty_work_disabled(app, catalog):
    model, tree = model_for(catalog)
    work = next(w for w in catalog.works if w.track_ids)
    row = next(i for i, w in enumerate(catalog.works) if w.work_id == work.work_id)
    index = tree.index(row, 0)
    assert tree.flags(index) & Qt.ItemFlag.ItemIsUserCheckable
    assert tree.setData(index, Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    assert work.work_id not in model.selected_work_ids
    track = tree.index(0, 0, index)
    assert not tree.flags(track) & Qt.ItemFlag.ItemIsUserCheckable
    assert track.data(Qt.ItemDataRole.CheckStateRole) is None
    assert not tree.setData(track, Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
    empty = next(i for i, w in enumerate(catalog.works) if not w.track_ids)
    assert not tree.flags(tree.index(empty, 0)) & Qt.ItemFlag.ItemIsUserCheckable
    model.begin_run(catalog, str(uuid4()))
    assert not model.set_work_selected(work.work_id, True)


def test_v05_filter_and_collapse_preserve_selection(app, catalog):
    model, tree = model_for(catalog)
    proxy = WorkFilterProxyModel()
    proxy.setSourceModel(tree)
    before = set(model.selected_work_ids)
    proxy.set_query("voice02")
    assert proxy.rowCount() == 1
    proxy.set_status_filter("selected")
    assert set(model.selected_work_ids) == before
    proxy.set_query("nonexistent")
    assert proxy.rowCount() == 0
    proxy.set_query("")
    assert set(model.selected_work_ids) == before


def test_v07_native_asr_values_and_v08_translation_indeterminate(app, catalog):
    model, tree = model_for(catalog)
    model.begin_run(catalog, str(uuid4()))
    track = catalog.planned_tracks[0]
    event(model, track, "stage_changed", stage="asr", payload={"progress_mode": "indeterminate"})
    assert model.tracks[track.track_id].ratio is None
    event(model, track, "asr_progress", revision=2, stage="asr", payload={"reported_ratio": .73,
        "current_audio_s": 7.3, "total_audio_s": 10., "source": "xxl_segment_start"})
    index = track_index(tree, track.track_id)
    assert index.data(ROLE_NODE)["progress"]["mode"] == "reported"
    assert model.tracks[track.track_id].ratio == .73
    event(model, track, "asr_progress", revision=3, stage="asr", payload={"reported_ratio": 1.,
        "current_audio_s": 10., "total_audio_s": 10., "source": "xxl_segment_start"})
    assert model.summary()["resolved"] == 0
    event(model, track, "stage_changed", revision=4, stage="translation", payload={"progress_mode": "indeterminate"})
    node = index.data(ROLE_NODE)
    assert node["progress"]["mode"] == "indeterminate"
    assert node["progress"]["ratio"] is None
    assert model.tracks[track.track_id].ratio is None


def test_v09_failures_and_not_started_denominator(app, catalog):
    model, tree = model_for(catalog)
    model.begin_run(catalog, str(uuid4()))
    work = next(w for w in catalog.works if len(w.track_ids) == 2)
    first, second = [catalog.track_index[tid] for tid in work.track_ids]
    event(model, first, "track_failed", status="failed", payload={"error": {"code": "TEST_FAILURE"}})
    event(model, second, "track_not_started", status="not_started", payload={"reason": "dependency_failed"})
    summary = model.work_summary(work.work_id)
    assert summary["planned"] == 2 and summary["resolved"] == 1
    assert summary["tone"] != "success"
    assert model.summary()["failed"] == 1 and model.summary()["not_started"] == 1
    assert track_index(tree, second.track_id).data(ROLE_NODE)["status"] == "not_started"


def test_v10_duplicate_dependency_and_representative_survive_sort(app, tmp_path):
    plan = build_execution_plan(library(tmp_path, ("RJ001/a/voice01.flac", "RJ001/b/voice01 SEなし.wav")))
    model, tree = model_for(plan)
    dependent = next(t for t in plan.tracks if t.dependencies)
    representative = dependent.representative_track_id
    index = track_index(tree, dependent.track_id)
    tooltip = index.data(Qt.ItemDataRole.ToolTipRole)
    assert tooltip and (representative in tooltip or "代表" in tooltip or "depend" in tooltip.lower())
    proxy = WorkFilterProxyModel()
    proxy.setSourceModel(tree)
    proxy.sort(0, Qt.SortOrder.DescendingOrder)
    assert plan.track_index[dependent.track_id].representative_track_id == representative
    assert model.plan is plan
    model.begin_run(plan, str(uuid4()))
    event(model, dependent, "track_finished", status="copied", payload={"outcome": "copied"})
    node = track_index(tree, dependent.track_id).data(ROLE_NODE)
    assert node["status"] == "copied" and node["progress"]["mode"] == "terminal"
    assert model.summary()["copied"] == 1


def test_model_rejects_stale_run_and_revision(app, catalog):
    model, _ = model_for(catalog)
    model.begin_run(catalog, str(uuid4()))
    track = catalog.planned_tracks[0]
    event(model, track, "track_started", revision=2)
    event(model, track, "track_failed", revision=1, status="failed")
    assert model.tracks[track.track_id].status == "running"
    event(model, track, "track_failed", revision=3, status="failed", run_id=str(uuid4()))
    assert model.tracks[track.track_id].status == "running"


def test_skipped_legacy_warning_is_visible(app, tmp_path):
    root = library(tmp_path, ("RJ001/voice01.wav",))
    (root / "RJ001/voice01.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nExisting legacy output.\n", encoding="utf-8")
    plan = build_execution_plan(root)
    model, tree = model_for(plan)
    track = plan.tracks[0]
    assert track.verification == "unverified_legacy"
    assert "unverified_legacy" in model.tracks[track.track_id].warnings
    assert "unverified_legacy" in track_index(tree, track.track_id).data(Qt.ItemDataRole.ToolTipRole)


def test_v17_settings_draft_ok_cancel_reset(app):
    from scripts.asmr_gui.settings import SettingsDialog, default_settings
    defaults = default_settings()
    assert {key: defaults[key] for key in ("parallel", "transcribe_parallel", "transcribe_batch_size", "translate_chunk_size")} == {
        "parallel": 5, "transcribe_parallel": 1, "transcribe_batch_size": 12, "translate_chunk_size": 20}
    assert defaults["fast_transcribe"] is True
    assert defaults["keep_intermediate_files"] is False
    original = dict(defaults, parallel=8)
    dialog = SettingsDialog(original, mode="transcribe_only", limit=2)
    dialog.fields["parallel"].setValue(3)
    dialog.fields["keep_intermediate_files"].setChecked(True)
    assert dialog.values()[0]["parallel"] == 3
    dialog.reset_button.click()
    assert dialog.values()[0]["parallel"] == 5
    assert dialog.values()[0]["keep_intermediate_files"] is False
    assert original["parallel"] == 8
    dialog.reject()
    assert original["parallel"] == 8
    accepted = SettingsDialog(original, mode="transcribe_only", limit=2)
    accepted.fields["parallel"].setValue(4)
    accepted.fields["keep_intermediate_files"].setChecked(True)
    accepted.accept()
    values, mode, limit = accepted.values()
    assert values["parallel"] == 4 and mode == "transcribe_only" and limit == 2
    assert values["keep_intermediate_files"] is True
    assert original["keep_intermediate_files"] is False
    assert original["parallel"] == 8


def window_for(tmp_path, catalog):
    from scripts.asmr_gui.controller import Controller
    from scripts.asmr_gui.window import BatchWindow
    controller = Controller(state_root=tmp_path / "state", project_root=PROJECT)
    controller.run_model.load_plan(catalog)
    controller.root = str(catalog.root)
    window = BatchWindow(controller)
    return controller, window


def test_v18_v21_minimum_controls_log_and_bound(app, tmp_path, catalog):
    controller, window = window_for(tmp_path, catalog)
    window.resize(980, 600)
    window.show()
    try:
        app.processEvents()
        for control in (window.start_button, window.stop_button, window.scan_button, window.settings_button):
            assert control.isVisible()
            assert window.rect().contains(control.mapTo(window, control.rect().topLeft()))
            assert window.rect().contains(control.mapTo(window, control.rect().bottomRight()))
        original_height = window.tree.height()
        window.log_toggle.click()
        app.processEvents()
        window.log_toggle.click()
        app.processEvents()
        assert abs(window.tree.height() - original_height) <= 2
        for number in range(1700):
            controller.log_message.emit(f"synthetic diagnostic {number}")
        app.processEvents()
        assert window.log_view.document().blockCount() <= 1500
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_v19_long_japanese_titles_tooltip(app, tmp_path):
    title = "長い日本語の作品タイトル" * 12
    plan = build_execution_plan(library(tmp_path, (title + "/voice01.wav",)))
    _, tree = model_for(plan)
    index = tree.index(0, 0)
    assert title in index.data(Qt.ItemDataRole.ToolTipRole)
    assert index.data()


def test_v20_frameless_maximize_restore_qt_state(app, tmp_path, catalog):
    _, window = window_for(tmp_path, catalog)
    window.show()
    try:
        app.processEvents()
        assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
        QTest.mouseDClick(window.title_bar, Qt.MouseButton.LeftButton, pos=QPoint(300, 20))
        app.processEvents()
        assert window.isMaximized()
        QTest.mouseDClick(window.title_bar, Qt.MouseButton.LeftButton, pos=QPoint(300, 20))
        app.processEvents()
        assert not window.isMaximized()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_v22_large_catalog_has_bounded_widgets(app, tmp_path):
    small = build_execution_plan(library(tmp_path, ("RJ001/voice01.wav",)))
    controller, window = window_for(tmp_path, small)
    before = len(window.findChildren(QWidget))
    work, track = small.works[0], small.tracks[0]
    tracks = tuple(replace(track, track_id=str(uuid4()), ordinal=i) for i in range(10000))
    large = replace(small, works=(replace(work, track_ids=tuple(t.track_id for t in tracks)),), tracks=tracks,
        duplicate_groups=(), excluded_track_ids=())
    started = time.monotonic()
    controller.run_model.load_plan(large)
    app.processEvents()
    assert time.monotonic() - started < 5
    assert window.tree_model.rowCount(window.tree_model.index(0, 0)) == 10000
    assert len(window.findChildren(QWidget)) <= before + 5
    assert len(window.findChildren(QProgressBar)) == 0
    source_index = window.tree_model.index(0, 2, window.tree_model.index(0, 0))
    assert window.tree.indexWidget(window.proxy.mapFromSource(source_index)) is None
    window.close()
    window.deleteLater()
    app.processEvents()


def test_v23_backend_independence_and_transcribe_only(app, tmp_path, catalog):
    controller, window = window_for(tmp_path, catalog)
    try:
        checks = {"available": False, "inference_proven": False,
            "asr": {"available": True, "required": True, "backend": "xxl", "device": "cuda"},
            "translation": {"available": False, "required": True, "error_code": "SAKURA_UNREACHABLE"}}
        controller.last_backend = checks
        controller.backend_changed.emit(checks)
        assert window.asr_label.text() == "ASR · XXL | 本機可用 · CUDA"
        assert "檢查失敗" in window.translation_label.text()
        assert "SAKURA_UNREACHABLE" in window.translation_label.toolTip()
        assert "SAKURA_UNREACHABLE" not in window.asr_label.text()
        assert "SAKURA_UNREACHABLE" not in window.asr_label.toolTip()
        assert "推論：尚未驗證" in window.asr_label.toolTip()
        assert "推論：尚未驗證" in window.translation_label.toolTip()
        assert checks["inference_proven"] is False
        assert controller.apply_settings({}, mode="transcribe_only")
        assert "不需要" in window.translation_label.text()
        assert not controller.last_backend
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_v12_generation_configuration_and_operation_identity_reject_old_callbacks(app, tmp_path, catalog):
    from scripts.asmr_gui.controller import Controller
    controller = Controller(state_root=tmp_path / "state")
    controller.set_root(str(catalog.root))
    controller.run_model.load_plan(catalog)
    controller._op_id = "old-operation"
    generation = controller._generation
    try:
        draft = {"parallel": 3}
        assert controller.apply_settings(draft)
        draft["parallel"] = 9
        assert controller.settings["parallel"] == 3
        controller._delivery.emit((generation, "old-operation", "phase", "RUNNING"))
        controller._delivery.emit((controller._generation, "other-operation", "backend", {"available": True}))
        controller._delivery.emit((generation, "old-operation", "catalog", (catalog, tmp_path)))
        app.processEvents()
        assert controller.state == "IDLE"
        assert not controller.last_backend
        assert controller.last_run_dir is None
        controller.run_model.set_work_selected(next(iter(controller.run_model.selected_work_ids)), False)
        assert controller._generation > generation + 1
    finally:
        finish_controller(app, controller)


def test_run_model_rejects_background_mutation(app, catalog):
    model, _ = model_for(catalog)
    failures = []

    def mutate():
        try:
            model.set_work_selected(next(iter(model.selected_work_ids)), False)
        except RuntimeError as exc:
            failures.append(str(exc))

    worker = threading.Thread(target=mutate)
    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert failures and "GUI thread" in failures[0]


def synthetic_operation_factory(operation, state_root, command_factory, **kwargs):
    """Use real read-only helper containment with test-only backend adapters."""
    from scripts.asmr_gui_runtime import ManagedOperation

    def command(managed):
        argv = command_factory(managed)
        return [str(DRIVER) if str(arg).replace("\\", "/").endswith("scripts/asmr_gui_worker.py") else arg for arg in argv]

    return ManagedOperation(operation, state_root, command, **kwargs)


def synthetic_worker_command(managed):
    from scripts.asmr_transcribe_lib.gui_plan_io import encode_plan
    document = encode_plan(managed.plan)
    return [sys.executable, "-B", "-u", str(DRIVER), "run",
        "--run-id", managed.run_id, "--job-name", managed.job.name,
        "--state-root", str(managed.state_root), "--project-root", str(PROJECT),
        "--plan", str(managed.run_dir / "plan.json"),
        "--expected-digest", document["transport_hash"]]


def actual_controller(tmp_path, root, **kwargs):
    from scripts.asmr_gui.controller import Controller
    return Controller(state_root=tmp_path / "state", project_root=PROJECT,
        python_executable=sys.executable, operation_factory=synthetic_operation_factory,
        worker_command=synthetic_worker_command, **kwargs)


def scan_ready(app, controller, root):
    assert controller.set_root(str(root))
    assert controller.scan()
    wait_for(app, lambda: controller.state in {"READY", "ERROR"}, 20)
    assert controller.state == "READY", controller.last_error


def finish_controller(app, controller):
    if controller.busy or controller.state == "CLEANUP_FAILED":
        controller.stop()
        wait_for(app, lambda: not controller.busy and controller.state != "CLEANUP_FAILED", 20)
    assert controller.request_close()


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
def test_v11_v13_v24_repeated_start_responsive_stop_close_guard(app, tmp_path):
    from scripts.asmr_gui_runtime import ManagedWorker
    root = library(tmp_path, ("RJ001/slow01.wav", "RJ001/slow02.wav"))
    workers, ticks = [], []

    def worker_factory(*args, **kwargs):
        worker = ManagedWorker(*args, cooperative_timeout=.15, cleanup_timeout=3, **kwargs)
        workers.append(worker)
        return worker

    controller = actual_controller(tmp_path, root, worker_factory=worker_factory)
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start(10)
    try:
        scan_ready(app, controller, root)
        assert controller.start()
        assert not controller.start()
        wait_for(app, lambda: controller.state in {"RUNNING", "ERROR"}, 20)
        assert controller.state == "RUNNING", controller.last_error
        wait_for(app, lambda: any(s.stage == "asr" for s in controller.run_model.tracks.values()), 10)
        assert len(workers) == 1
        close_ready = QSignalSpy(controller.close_ready)
        started, before = time.monotonic(), len(ticks)
        assert controller.request_close() is False
        assert time.monotonic() - started < .25
        assert not controller.start() and not controller.clear()
        wait_for(app, lambda: len(close_ready) == 1, 15)
        assert len(ticks) > before
        cleanup = json.loads((workers[0].run_dir / "cleanup.json").read_text(encoding="utf-8"))
        assert cleanup["cleanup_verified"] and cleanup["active_count"] == 0
        assert workers[0].job.handle is None
        assert controller.request_close()
    finally:
        timer.stop()
        finish_controller(app, controller)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
def test_v15_v16_missing_live_events_reconcile_once(app, tmp_path):
    from scripts.asmr_gui_runtime import ManagedWorker
    root = library(tmp_path, ("RJ001/voice01.wav", "RJ001/voice02.wav", "RJ002/voice03.wav"))
    workers = []

    def muted_worker(*args, **kwargs):
        kwargs["on_stdout"] = lambda chunk: None
        worker = ManagedWorker(*args, **kwargs)
        workers.append(worker)
        return worker

    controller = actual_controller(tmp_path, root, worker_factory=muted_worker)
    completed = QSignalSpy(controller.completed)
    try:
        scan_ready(app, controller, root)
        selected = next(w for w in controller.run_model.plan.works if w.relative_path == "RJ001")
        excluded = next(w for w in controller.run_model.plan.works if w.relative_path == "RJ002")
        assert controller.run_model.set_work_selected(excluded.work_id, False)
        assert controller.start()
        wait_for(app, lambda: len(completed) == 1, 25)
        assert controller.state == "FINISHED", controller.last_error
        assert controller.run_model.summary()["success"] == 2
        assert controller.run_model.work_summary(selected.work_id)["resolved"] == 2
        assert workers[0].plan.selected_work_ids == (selected.work_id,)
        assert len(workers[0].plan.planned_tracks) == 2
        assert all(controller.run_model.tracks[tid].status == "success" for tid in selected.track_ids)
        assert all(controller.run_model.tracks[tid].status != "success" for tid in excluded.track_ids)
        assert json.loads((workers[0].run_dir / "cleanup.json").read_text(encoding="utf-8"))["cleanup_verified"]
        wait_for(app, lambda: not controller.busy)
        controller._delivery.emit((controller._generation, controller._op_id, "finished",
            (controller.last_cleanup, completed[0][0], controller.last_run_dir)))
        for _ in range(20):
            app.processEvents()
        assert len(completed) == 1
    finally:
        finish_controller(app, controller)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
def test_v15_live_sequence_gap_recovers_active_stage(app, tmp_path):
    from scripts.asmr_gui_runtime import ManagedWorker
    root = library(tmp_path, ("RJ001/slow01.wav",))
    dropped, workers = [], []

    def gapped_worker(*args, **kwargs):
        original = kwargs["on_stdout"]
        pending = bytearray()

        def stdout(data):
            pending.extend(data)
            while b"\n" in pending:
                line, _, tail = pending.partition(b"\n")
                pending[:] = tail
                frame = json.loads(line)
                if not dropped and frame["event"] == "track_started":
                    dropped.append(frame["seq"])
                    continue
                original(bytes(line) + b"\n")

        kwargs["on_stdout"] = stdout
        worker = ManagedWorker(*args, cooperative_timeout=.15, cleanup_timeout=3, **kwargs)
        workers.append(worker)
        return worker

    controller = actual_controller(tmp_path, root, worker_factory=gapped_worker)
    logs = QSignalSpy(controller.log_message)
    try:
        scan_ready(app, controller, root)
        assert controller.start()
        wait_for(app, lambda: bool(dropped), 15)
        wait_for(app, lambda: any(s.status == "running" and s.stage == "asr" for s in controller.run_model.tracks.values()), 4)
        assert controller.state == "RUNNING"
        assert workers[0].process.poll() is None
        assert any("事件" in str(entry[0]) for entry in logs)
        assert controller.stop()
        wait_for(app, lambda: not controller.busy, 15)
        assert controller.last_cleanup["cleanup_verified"]
    finally:
        finish_controller(app, controller)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
def test_v09_authoritative_failed_receipt_never_becomes_success(app, tmp_path):
    root = library(tmp_path, ("RJ001/fail01.wav",))
    controller = actual_controller(tmp_path, root)
    completed = QSignalSpy(controller.completed)
    try:
        scan_ready(app, controller, root)
        assert controller.start()
        wait_for(app, lambda: len(completed) == 1, 25)
        result = completed[0][0]
        assert result["outcome"] == "completed_with_errors"
        assert result["counts"]["failed"] == 1
        assert controller.run_model.summary()["resolved"] == 1
        assert controller.run_model.summary()["success"] == 0
        work = controller.run_model.plan.works[0]
        summary = controller.run_model.work_summary(work.work_id)
        assert summary["progress"]["ratio"] == 1
        assert summary["tone"] == "danger"
        track = controller.run_model.plan.tracks[0]
        receipt = json.loads((controller.last_run_dir / "receipts" / (track.track_id + ".json")).read_text(encoding="utf-8"))
        assert receipt["status"] == "failed" and receipt["provenance"] is None
        assert controller.last_cleanup["cleanup_verified"]
    finally:
        finish_controller(app, controller)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
def test_current_session_trusted_transcription_receipts_enable_resume(app, tmp_path):
    """A rescan may reuse only completed, validated current-session outputs."""
    from scripts.asmr_gui_runtime import ManagedWorker
    from scripts.asmr_transcribe_lib.gui_protocol import JSONLDecoder
    from scripts.asmr_transcribe_lib.gui_results import resume_provenance
    root = library(tmp_path, ("RJ001/voice01.wav", "RJ001/nested/voice02.wav"))
    workers = []

    def worker_factory(*args, **kwargs):
        worker = ManagedWorker(*args, **kwargs)
        workers.append(worker)
        return worker

    controller = actual_controller(tmp_path, root, worker_factory=worker_factory)
    completed = QSignalSpy(controller.completed)
    try:
        assert controller.apply_settings({}, mode="transcribe_only")
        scan_ready(app, controller, root)
        assert controller.start()
        wait_for(app, lambda: len(completed) == 1, 25)
        assert controller.state == "FINISHED", controller.last_error
        first = workers[0]
        assert first.plan.mode == "transcribe_only"
        assert controller.last_cleanup["cleanup_verified"]
        assert completed[0][0]["counts"]["transcribed"] == 2
        proofs = resume_provenance(first.run_dir, first.run_id, first.plan.plan_hash)
        assert len(proofs) == 2 and {proof.output_type for proof in proofs} == {"transcription"}
        first_events = JSONLDecoder(first.run_id).feed(first.stdout_bytes)
        assert any(e["event"] == "stage_changed" and e["stage"] == "asr" for e in first_events)
        sources = {track.task.audio_path: track.task.audio_path.read_bytes() for track in first.plan.planned_tracks}
        outputs = {track.task.transcribe_srt_path: track.task.transcribe_srt_path.read_bytes() for track in first.plan.planned_tracks}

        # Exercise the public scan path and real read-only helper. Tests neither
        # install proof into the controller nor rebuild the displayed plan.
        assert controller.scan()
        wait_for(app, lambda: controller.state in {"READY", "ERROR"}, 20)
        assert controller.state == "READY", controller.last_error
        resumed = controller.run_model.plan
        assert resumed.executable, [issue.code for issue in resumed.issues]
        assert "UNVERIFIED_LEGACY" not in {issue.code for issue in resumed.issues}
        assert all(track.verification == "verified" for track in resumed.planned_tracks)
        assert controller.start()
        wait_for(app, lambda: len(completed) == 2, 25)
        assert controller.state == "FINISHED", controller.last_error
        second = workers[1]
        assert second.run_id != first.run_id
        assert second.plan.settings_hash == first.plan.settings_hash
        assert completed[1][0]["counts"]["transcribed"] == 2
        assert controller.run_model.summary()["transcribed"] == 2
        assert controller.last_cleanup["cleanup_verified"]
        events = JSONLDecoder(second.run_id).feed(second.stdout_bytes)
        assert not any(e["stage"] == "asr" or e["event"] == "asr_progress" for e in events)
        reused = {e["track_id"] for e in events if e["event"] == "stage_changed" and e["payload"].get("reused")}
        assert reused == {track.track_id for track in second.plan.planned_tracks}
        assert all(path.read_bytes() == contents for path, contents in sources.items())
        assert all(path.read_bytes() == contents for path, contents in outputs.items())
    finally:
        finish_controller(app, controller)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
def test_unreceipted_legacy_transcription_is_blocked_by_controller(app, tmp_path):
    from scripts.asmr_gui_runtime import ManagedWorker
    root = library(tmp_path, ("RJ001/voice01.wav",))
    plan = build_execution_plan(root, mode="transcribe_only")
    output = plan.tracks[0].task.transcribe_srt_path
    legacy = "1\n00:00:00,000 --> 00:00:01,000\nLegacy text without a trusted stage receipt.\n"
    output.write_text(legacy, encoding="utf-8")
    original = output.read_bytes()
    workers = []

    def worker_factory(*args, **kwargs):
        workers.append(True)
        return ManagedWorker(*args, **kwargs)

    controller = actual_controller(tmp_path, root, worker_factory=worker_factory)
    try:
        assert controller.apply_settings({}, mode="transcribe_only")
        scan_ready(app, controller, root)
        assert "UNVERIFIED_LEGACY" in {issue.code for issue in controller.run_model.plan.issues}
        assert not controller.run_model.plan.executable
        if controller.start():
            wait_for(app, lambda: not controller.busy, 20)
            assert controller.state == "ERROR"
        assert not workers
        assert output.read_bytes() == original
        assert not plan.tracks[0].task.final_srt_path.exists()
    finally:
        finish_controller(app, controller)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
def test_unselected_invalid_work_does_not_block_fresh_selected_work(app, tmp_path):
    from scripts.asmr_gui_runtime import ManagedWorker
    root = library(tmp_path, ("WorkA/voice01.wav", "WorkB/voice02.wav"))
    initial = build_execution_plan(root, mode="transcribe_only")
    invalid = next(track for track in initial.tracks if track.task.work_dir.name == "WorkA")
    legacy = invalid.task.transcribe_srt_path
    legacy.write_text("1\n00:00:00,000 --> 00:00:01,000\nUnreceipted legacy text in an excluded Work.\n", encoding="utf-8")
    legacy_bytes = legacy.read_bytes()
    source_bytes = invalid.task.audio_path.read_bytes()
    workers = []

    def worker_factory(*args, **kwargs):
        worker = ManagedWorker(*args, **kwargs)
        workers.append(worker)
        return worker

    controller = actual_controller(tmp_path, root, worker_factory=worker_factory)
    completed = QSignalSpy(controller.completed)
    try:
        assert controller.apply_settings({}, mode="transcribe_only")
        scan_ready(app, controller, root)
        catalog = controller.run_model.plan
        work_a = next(work for work in catalog.works if work.relative_path == "WorkA")
        work_b = next(work for work in catalog.works if work.relative_path == "WorkB")
        assert not catalog.executable
        assert any(issue.code == "UNVERIFIED_LEGACY" and invalid.track_id in issue.track_ids for issue in catalog.issues)
        assert controller.run_model.set_work_selected(work_a.work_id, False)
        assert controller.run_model.selected_work_ids == {work_b.work_id}
        assert controller.start()
        wait_for(app, lambda: len(completed) == 1 or controller.state == "ERROR", 25)
        assert controller.state == "FINISHED", controller.last_error
        assert len(workers) == 1
        assert workers[0].plan.selected_work_ids == (work_b.work_id,)
        assert workers[0].plan.executable
        assert completed[0][0]["counts"]["transcribed"] == 1
        assert controller.run_model.summary()["transcribed"] == 1
        assert controller.last_cleanup["cleanup_verified"]
        assert legacy.read_bytes() == legacy_bytes
        assert invalid.task.audio_path.read_bytes() == source_bytes
        assert not invalid.task.final_srt_path.exists()
    finally:
        finish_controller(app, controller)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
def test_curated_diagnostics_are_durable_private_and_written_off_gui_thread(app, tmp_path, monkeypatch):
    from PyQt6.QtGui import QDesktopServices
    from scripts.asmr_gui.window import BatchWindow
    root = library(tmp_path, ("RJ001/fail01.wav",))
    controller = actual_controller(tmp_path, root)
    completed = QSignalSpy(controller.completed)
    diagnostic = controller.diagnostic_log_path
    assert not diagnostic.resolve().is_relative_to(root.resolve())
    original_open = Path.open
    writes = []

    def tracked_open(path, mode="r", *args, **kwargs):
        if path == diagnostic and any(flag in mode for flag in "wax"):
            writes.append((threading.get_ident(), threading.current_thread().name))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)

    def text():
        try:
            return diagnostic.read_text(encoding="utf-8")
        except OSError:
            return ""

    window = None
    try:
        scan_ready(app, controller, root)
        assert controller.start()
        wait_for(app, lambda: len(completed) == 1, 25)
        assert completed[0][0]["outcome"] == "completed_with_errors"
        wait_for(app, lambda: all(code in text() for code in (
            "lifecycle · SCANNING", "lifecycle · RUNNING", "lifecycle · RECONCILING",
            "lifecycle · FINISHED", "track_failed", "completed_with_errors", "cleanup · verified=True active=0")))

        def failing_operation(*args, **kwargs):
            raise RuntimeError("PRIVATE_TRANSCRIPTION_SENTINEL token=PRIVATE_SECRET")

        controller.operation_factory = failing_operation
        assert controller.scan()
        wait_for(app, lambda: controller.state == "ERROR")
        wait_for(app, lambda: "SCAN_FAILED:RuntimeError" in text())
        content = text()
        assert "lifecycle · ERROR" in content
        assert "PRIVATE" not in content and "token=" not in content
        assert writes and all(identity != threading.get_ident() for identity, _ in writes)
        assert all(name == "asmr-diagnostics" for _, name in writes)

        # The full-log action resolves this durable file rather than a new or
        # reconstructed protocol journal. Do not open an external application.
        opened = []
        monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
        window = BatchWindow(controller)
        window.journal_button.click()
        assert [Path(path).resolve() for path in opened] == [diagnostic.resolve()]
    finally:
        if window is not None:
            window.close()
            window.deleteLater()
            app.processEvents()
        else:
            finish_controller(app, controller)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
def test_v12_cancelled_preflight_cannot_launch_stale_worker(app, tmp_path, monkeypatch):
    from scripts.asmr_gui_runtime import ManagedWorker
    root = library(tmp_path)
    workers = []

    def worker_factory(*args, **kwargs):
        workers.append(True)
        return ManagedWorker(*args, **kwargs)

    monkeypatch.setenv("ASMR_GUI_TEST_PREFLIGHT_DELAY", "1.5")
    controller = actual_controller(tmp_path, root, worker_factory=worker_factory)
    try:
        scan_ready(app, controller, root)
        assert controller.start()
        wait_for(app, lambda: controller.state == "PREFLIGHT", 10)
        assert controller.stop()
        wait_for(app, lambda: not controller.busy, 20)
        deadline = time.monotonic() + 1.8
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert not workers
        assert controller.state != "RUNNING"
    finally:
        finish_controller(app, controller)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
def test_v14_cleanup_failure_holds_guard_until_verified_retry(app, tmp_path):
    from scripts.asmr_gui_runtime import ManagedWorker
    root = library(tmp_path, ("RJ001/slow01.wav",))
    workers = []

    class FirstCleanupFails(ManagedWorker):
        def wait(self, timeout=None):
            if self.process is not None and self.process.poll() is None:
                return {"state": "cleanup_failed", "run_id": self.run_id,
                    "cleanup_verified": False, "error": "SYNTHETIC_QUERY_FAILURE"}
            return super().wait(timeout)

    def worker_factory(*args, **kwargs):
        worker = FirstCleanupFails(*args, cooperative_timeout=.15, cleanup_timeout=3, **kwargs)
        workers.append(worker)
        return worker

    controller = actual_controller(tmp_path, root, worker_factory=worker_factory)
    try:
        scan_ready(app, controller, root)
        assert controller.start()
        wait_for(app, lambda: controller.state in {"CLEANUP_FAILED", "ERROR"}, 20)
        assert controller.state == "CLEANUP_FAILED", controller.last_error
        assert workers[0].job.handle is not None
        assert not controller.start() and not controller.clear()
        assert controller.request_close() is False
        assert controller.stop()
        wait_for(app, lambda: controller.state != "CLEANUP_FAILED" and not controller.busy, 20)
        assert workers[0].job.handle is None
        assert controller.request_close()
    finally:
        finish_controller(app, controller)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
def test_evidence_host_control_and_geometry_offscreen(app, tmp_path):
    """Verify fixture wiring; this does not replace physical Windows captures."""
    evidence = tmp_path / "evidence"
    control = tmp_path / "control.json"
    host = PROJECT / "tests" / "fixtures" / "asmr_gui_v1_visual.py"
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    process = subprocess.Popen([sys.executable, "-B", "-u", str(host),
        "--state-root", str(tmp_path / "state"), "--library", str(tmp_path / "library"),
        "--control-file", str(control), "--evidence-dir", str(evidence)],
        cwd=str(PROJECT), env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def latest():
        try:
            return json.loads((evidence / "latest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    try:
        wait_for(app, lambda: latest().get("state") in {"READY", "ERROR"} or process.poll() is not None, 20)
        if process.poll() is not None:
            raise AssertionError(process.communicate(timeout=1))
        document = latest()
        assert document["state"] == "READY", document
        assert document["synthetic"] and document["inference_proven"] is False
        assert document["geometry"]["qt_geometry"]["width"] == 1220
        assert document["geometry"]["qt_frame_geometry"]["width"] > 0
        assert document["controls"]["stop_button"]["visible"]
        assert {work["kind"] for work in document["works"]} == {"directory", "synthetic_root"}
        control.write_text(json.dumps({"request_id": "host-smoke-resize", "action": "resize", "width": 980, "height": 600}), encoding="utf-8")
        wait_for(app, lambda: latest().get("geometry", {}).get("qt_geometry", {}).get("width") == 980)
        assert all(button["top_left_inside"] and button["bottom_right_inside"] for button in latest()["controls"].values())
        capture = evidence / "offscreen-own-window.png"
        control.write_text(json.dumps({"request_id": "host-smoke-capture", "action": "capture", "path": str(capture)}), encoding="utf-8")
        wait_for(app, lambda: capture.is_file())
        wait_for(app, lambda: latest().get("capture") is not None)
        assert latest()["capture"]["sample_colors"] >= 8
        assert latest()["capture"]["native_visible"] is False
        control.write_text(json.dumps({"request_id": "host-smoke-layout", "action": "layout_audit"}), encoding="utf-8")
        wait_for(app, lambda: (evidence / "layout-audit.json").is_file())
        audit = json.loads((evidence / "layout-audit.json").read_text(encoding="utf-8"))
        assert len(audit["observations"]) == 3
        assert all(row["all_essential_inside"] for row in audit["observations"])
        control.write_text(json.dumps({"request_id": "host-smoke-quit", "action": "quit"}), encoding="utf-8")
        process.wait(timeout=15)
        assert process.returncode == 0, process.communicate(timeout=1)
    finally:
        if process.poll() is None:
            control.write_text(json.dumps({"request_id": "host-finally-quit", "action": "quit"}), encoding="utf-8")
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()


@pytest.mark.parametrize("saved", [{}, {"parallel": 3}, {"keep_intermediate_files": True},
                                   {"keep_intermediate_files": "false"}])
def test_cleanup_saved_settings_compatibility(app, tmp_path, catalog, saved):
    from scripts.asmr_gui.window import QSettings
    from scripts.asmr_gui.settings import default_settings
    preferences = QSettings("VoiceTransl", "ASMRBatchGUI")
    preferences.setValue("batch/settings", json.dumps(saved))
    controller, window = window_for(tmp_path, catalog)
    try:
        effective = {**default_settings(), **controller.settings}
        assert effective["keep_intermediate_files"] is (saved.get("keep_intermediate_files") is True)
        assert type(effective["keep_intermediate_files"]) is bool
        if saved.get("parallel"):
            assert effective["parallel"] == 3
    finally:
        window._close_authorized = True
        window.close()
        controller.request_close()


@pytest.mark.parametrize("accept", [False, True])
def test_cleanup_settings_ok_persists_cancel_discards(app, tmp_path, catalog, monkeypatch, accept):
    from PyQt6.QtWidgets import QDialog
    from scripts.asmr_gui.settings import SettingsDialog
    controller, window = window_for(tmp_path, catalog)
    def execute(dialog):
        dialog.fields["keep_intermediate_files"].setChecked(True)
        return QDialog.DialogCode.Accepted if accept else QDialog.DialogCode.Rejected
    monkeypatch.setattr(SettingsDialog, "exec", execute)
    try:
        window._settings()
        assert controller.settings.get("keep_intermediate_files", False) is accept
        stored = json.loads(window.preferences.value("batch/settings", "{}"))
        assert stored.get("keep_intermediate_files", False) is accept
        if accept:
            assert controller.apply_settings({"keep_intermediate_files": "false"}) is False
            assert controller.settings["keep_intermediate_files"] is True
    finally:
        window._close_authorized = True
        window.close()
        controller.request_close()


def test_cleanup_dialog_rejects_malformed_setting(app):
    from scripts.asmr_gui.settings import SettingsDialog
    with pytest.raises(ValueError, match="boolean"):
        SettingsDialog({"keep_intermediate_files": "false"})


def test_cleanup_preferences_preserve_existing_mode_and_limit(app, tmp_path, catalog):
    from scripts.asmr_gui.controller import Controller
    from scripts.asmr_gui.window import BatchWindow, QSettings
    QSettings("VoiceTransl", "ASMRBatchGUI").setValue(
        "batch/settings", json.dumps({"keep_intermediate_files": True}))
    controller = Controller(state_root=tmp_path / "state", project_root=PROJECT)
    controller.mode, controller.limit = "transcribe_only", 2
    window = BatchWindow(controller)
    try:
        assert controller.mode == "transcribe_only" and controller.limit == 2
        assert controller.settings["keep_intermediate_files"] is True
    finally:
        window._close_authorized = True
        window.close()
        controller.request_close()
