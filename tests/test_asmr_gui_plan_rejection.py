"""Deterministic Qt diagnostics: rejected plans never reach a backend or owner."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QObject, QSettings, pyqtSignal
from PyQt6.QtTest import QSignalSpy

from scripts.asmr_gui.controller import Controller
from scripts.asmr_gui.diagnostics import plan_rejection_diagnostic
from scripts.asmr_transcribe_lib.gui_plan import PlanIssue, PlanRejected, build_execution_plan


class QuietHealth(QObject):
    updated = pyqtSignal(dict)

    def __init__(self, **kwargs):
        super().__init__(kwargs.get("parent"))

    def configure(self, *args):
        pass

    set_paused = configure
    close = configure


@pytest.fixture
def controller(tmp_path, qtbot):
    def forbidden(*args, **kwargs):
        raise AssertionError("rejected plan must not start preflight, inference or an owned worker")

    instance = Controller(state_root=tmp_path / "state", operation_factory=forbidden,
                          worker_factory=forbidden, health_monitor_factory=QuietHealth)
    yield instance
    assert not instance.busy
    assert instance.request_close()


def fake_scan(controller, qtbot, root, mode="full"):
    """Real planner catalog, deterministic delivery instead of a subprocess."""
    catalog = build_execution_plan(root, mode=mode, skip_existing_subtitles=True)
    controller.apply_settings({}, mode=mode)
    controller._scan = lambda intent: controller._post(intent, "catalog", (catalog, root, None))
    assert controller.scan(str(root))
    qtbot.waitUntil(lambda: controller.state == "READY")
    return catalog


def make_library(tmp_path, collision=True):
    root = tmp_path / "library"
    # Root-level Tracks deliberately keep separate groups (legacy policy), so
    # these remain unsafe even after same-group shared-output support.
    names = ["Track01.flac"]
    if collision:
        names.append("Track01.mp3")
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SYNTHETIC_AUDIO")
    return root


def start_rejected(controller, qtbot):
    signals = QSignalSpy(controller.log_message)
    assert controller.start()
    qtbot.waitUntil(lambda: controller.state == "ERROR")
    assert controller._future.result(timeout=5) is None
    assert controller._owned is None
    return [entry[0] for entry in signals]


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_same_stem_collision_has_bounded_gui_and_complete_local_evidence(controller, qtbot, tmp_path, mode, monkeypatch):
    import scripts.asmr_gui.window as window_module
    root = make_library(tmp_path)
    original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    catalog = fake_scan(controller, qtbot, root, mode)
    assert {i.code for i in catalog.issues} == {"OUTPUT_NAMESPACE_COLLISION"}
    assert len(catalog.issues) == (8 if mode == "full" else 6)
    monkeypatch.setattr(window_module, "QSettings", lambda *args: QSettings(
        str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat))
    window = window_module.BatchWindow(controller)
    qtbot.addWidget(window)
    lines = start_rejected(controller, qtbot)
    assert "OUTPUT_NAMESPACE_COLLISION" in window.log_view.toPlainText()
    assert "Traceback" not in window.log_view.toPlainText()
    errors = [line for line in lines if "RUN_FAILED:PlanRejected" in line]
    assert len(errors) == 1
    assert f"OUTPUT_NAMESPACE_COLLISION: tracks=2 issues={len(catalog.issues)}" in errors[0]
    assert "origin=plan_construction" in errors[0]
    assert str(root) not in errors[0] and "Traceback" not in errors[0]
    assert len(errors[0]) < 1024
    artifact = next((controller.state_root / "diagnostics").glob("*.json"))
    report = json.loads(artifact.read_text(encoding="utf-8"))
    affected = {key for issue in catalog.issues for key in issue.track_ids}
    assert {t["track_id"] for t in report["tracks"]} == affected
    assert {t["audio_path"] for t in report["tracks"]} == {
        str(catalog.track_index[key].task.audio_path) for key in affected}
    assert {p for i in report["issues"] for p in i["paths"]} == {
        str(p) for i in catalog.issues for p in i.paths}
    assert any(p.endswith(".tmp") for i in report["issues"] for p in i["paths"])
    assert report["affected_track_count"] == 2
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == original


def test_many_issues_keep_full_artifact_but_no_exception_or_settings_secrets(controller, qtbot, tmp_path):
    root = make_library(tmp_path)
    catalog = fake_scan(controller, qtbot, root)
    intent = replace(controller._new_intent("run"),
                     settings_json='{"secret": "DO_NOT_DISCLOSE_SETTINGS"}')
    base = catalog.issues[0]
    issues = tuple(replace(base, paths=(root / f"sample-{i:04}.srt",)) for i in range(1200))
    rejection = PlanRejected(issues)
    rejection.args = ("PRIVATE_TRANSCRIPT_DO_NOT_DISCLOSE",)
    report, summary = plan_rejection_diagnostic(intent, rejection)
    assert len(report["issues"]) == 1200
    assert len(report["tracks"]) == 2
    assert "issues=1200 tracks=2" in summary
    assert "sample-0004.srt" in summary and "sample-0005.srt" not in summary
    assert len(summary) < 1024
    assert "DO_NOT_DISCLOSE" not in json.dumps(report) + summary
    assert str(root) not in summary


def test_start_freshness_rejection_is_identified_without_changing_validation(controller, qtbot, tmp_path, monkeypatch):
    import scripts.asmr_gui.controller as module
    root = make_library(tmp_path, collision=False)
    catalog = fake_scan(controller, qtbot, root)
    original_build = module.build_execution_plan

    def change_after_build(*args, **kwargs):
        plan = original_build(*args, **kwargs)
        catalog.tracks[0].task.audio_path.write_bytes(b"CHANGED_SYNTHETIC_AUDIO")
        return plan

    monkeypatch.setattr(module, "build_execution_plan", change_after_build)
    start_rejected(controller, qtbot)
    assert "PLAN_STALE" in controller.last_error
    assert "origin=start_freshness_validation" in controller.last_error
    assert "tracks=1" in controller.last_error


def test_catalog_membership_rejection_has_its_own_boundary(controller, qtbot, tmp_path):
    root = make_library(tmp_path, collision=False)
    fake_scan(controller, qtbot, root)
    (root / "new.wav").write_bytes(b"ADDED_SYNTHETIC_AUDIO")
    start_rejected(controller, qtbot)
    assert "PLAN_STALE" in controller.last_error
    assert "origin=catalog_freshness_validation" in controller.last_error


def test_diagnostic_write_failure_still_rejects_plan(controller, qtbot, tmp_path, monkeypatch):
    import scripts.asmr_gui.controller as module
    fake_scan(controller, qtbot, make_library(tmp_path))

    def denied(*args, **kwargs):
        raise PermissionError("PRIVATE_WRITE_FAILURE")

    monkeypatch.setattr(module, "atomic_json", denied)
    start_rejected(controller, qtbot)
    assert "OUTPUT_NAMESPACE_COLLISION" in controller.last_error
    assert "diagnostic=unavailable" in controller.last_error
    assert "PRIVATE_WRITE_FAILURE" not in controller.last_error


def test_diagnostics_never_write_inside_library(controller, qtbot, tmp_path):
    root = make_library(tmp_path)
    fake_scan(controller, qtbot, root)
    controller.state_root = root / "must-not-exist"
    start_rejected(controller, qtbot)
    assert not controller.state_root.exists()
    assert "diagnostic=unavailable" in controller.last_error


def test_rejection_with_retained_owner_preserves_cleanup_failed_branch(controller, qtbot, tmp_path, monkeypatch):
    fake_scan(controller, qtbot, make_library(tmp_path))
    intent = controller._new_intent("run")
    owner = object()
    messages = []
    monkeypatch.setattr(controller, "_post", lambda *args: messages.append(args))
    controller._owned = owner
    try:
        controller._error(intent, "RUN_FAILED", PlanRejected(intent.catalog.issues))
        assert controller._owned is owner
        assert messages == [(intent, "cleanup_failed", {"cleanup_verified": False, "error": "RUN_FAILED"})]
        assert not (controller.state_root / "diagnostics").exists()
    finally:
        controller._owned = None


def test_other_exceptions_remain_curated(controller, qtbot, tmp_path, monkeypatch):
    fake_scan(controller, qtbot, make_library(tmp_path, collision=False))
    messages = []
    monkeypatch.setattr(controller, "_post", lambda *args: messages.append(args))
    intent = controller._new_intent("run")
    controller._error(intent, "RUN_FAILED", ValueError("PRIVATE_TRANSCRIPT_AND_SECRET"))
    assert messages == [(intent, "error", "RUN_FAILED:ValueError")]
