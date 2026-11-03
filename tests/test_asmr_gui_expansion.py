"""Work expansion is transient view state, separate from execution selection."""
from uuid import uuid4

import pytest
from PyQt6.QtCore import Qt

import test_asmr_gui_recovery as recovery

gui = recovery.gui


def make_library(root):
    return recovery.library(root, tuple(
        f"Work{work}/Track{track:02d}_{'needle' if work == 'B' and track == 17 else 'audio'}.wav"
        for work in ("A", "B") for track in range(24)))


def ready(qtbot, gui, tmp_path):
    make, preferences = gui
    root = make_library(tmp_path / "library")
    preferences.setValue("last_input_root", str(root))
    controller, window = make()
    qtbot.waitUntil(lambda: controller.state == "READY", timeout=5000)
    return root, controller, window, preferences


def expanded(window):
    return {window.proxy.index(row, 0).data() for row in range(window.proxy.rowCount())
            if window.tree.isExpanded(window.proxy.index(row, 0))}


def test_cold_start_all_48_tracks_collapsed(qtbot, gui, tmp_path):
    _, controller, window, _ = ready(qtbot, gui, tmp_path)
    assert len(controller.run_model.tracks) == 48
    assert window.proxy.rowCount() == 2
    assert expanded(window) == set()


def test_user_expansion_survives_progress_catalog_and_selection(qtbot, gui, tmp_path):
    _, controller, window, _ = ready(qtbot, gui, tmp_path)
    window.tree.collapseAll()  # Isolate preservation from the default-state regression.
    original = set(controller.run_model.selected_work_ids)
    first = window.proxy.index(0, 0)
    window.tree.expand(first)
    assert expanded(window) == {"WorkA"}
    assert controller.run_model.selected_work_ids == original
    track = controller.run_model.plan.works[0].track_ids[0]
    controller.run_model.changed.emit({"track_ids": (track,)})
    controller.catalog_changed.emit()
    assert expanded(window) == {"WorkA"}
    assert window.proxy.setData(first, Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    assert expanded(window) == {"WorkA"}
    assert len(controller.run_model.selected_work_ids) == 1
    window.tree.collapse(first)
    assert expanded(window) == set()
    assert len(controller.run_model.selected_work_ids) == 1


def test_same_root_rescan_retains_expansion(qtbot, gui, tmp_path):
    _, controller, window, _ = ready(qtbot, gui, tmp_path)
    window.tree.collapseAll()
    window.tree.expand(window.proxy.index(1, 0))
    with qtbot.waitSignal(controller.catalog_changed, timeout=5000):
        window.scan_button.click()
    assert expanded(window) == {"WorkB"}


def test_root_switch_starts_collapsed(qtbot, gui, tmp_path, monkeypatch):
    _, controller, window, _ = ready(qtbot, gui, tmp_path)
    window.tree.expand(window.proxy.index(0, 0))
    other = make_library(tmp_path / "other")
    monkeypatch.setattr(recovery.window_module.QFileDialog, "getExistingDirectory", lambda *args: str(other))
    with qtbot.waitSignal(controller.catalog_changed, timeout=5000,
                         check_params_cb=lambda: controller.state == "READY"):
        window.browse_button.click()
    assert expanded(window) == set()


@pytest.mark.parametrize("rescan", [False, True])
def test_track_search_and_clear_restores_user_expansion(qtbot, gui, tmp_path, rescan):
    _, controller, window, _ = ready(qtbot, gui, tmp_path)
    window.tree.collapseAll()
    window.tree.expand(window.proxy.index(0, 0))
    selected = set(controller.run_model.selected_work_ids)
    window.search_edit.setText("needle")
    assert window.proxy.rowCount() == 1
    assert expanded(window) == {"WorkB"}
    # Search expansion changes remain temporary, including explicit user toggles.
    window.tree.collapse(window.proxy.index(0, 0))
    if rescan:
        with qtbot.waitSignal(controller.catalog_changed, timeout=5000):
            window.scan_button.click()
    window.search_edit.clear()
    assert expanded(window) == {"WorkA"}
    assert controller.run_model.selected_work_ids == selected


def test_start_stop_model_reset_preserves_expansion(qtbot, gui, tmp_path):
    _, controller, window, _ = ready(qtbot, gui, tmp_path)
    window.tree.collapseAll()
    window.tree.expand(window.proxy.index(0, 0))
    plan = controller.run_model.plan
    try:
        controller.run_model.begin_run(plan, str(uuid4()))
        controller._set_state("RUNNING")
        assert expanded(window) == {"WorkA"}
        controller._set_state("STOPPING")
        assert expanded(window) == {"WorkA"}
        controller._set_state("FINISHED")
        assert expanded(window) == {"WorkA"}
    finally:
        controller._set_state("IDLE")


def test_expansion_not_persisted_in_preferences(qtbot, gui, tmp_path):
    _, _, window, preferences = ready(qtbot, gui, tmp_path)
    before = set(preferences.allKeys())
    window.tree.expand(window.proxy.index(0, 0))
    window.tree.collapse(window.proxy.index(1, 0))
    preferences.sync()
    assert set(preferences.allKeys()) == before
