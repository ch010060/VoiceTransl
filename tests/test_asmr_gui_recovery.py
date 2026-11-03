"""Recovery integration: deterministic Qt interaction and read-only history."""
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_QT_API", "pyqt6")
from PyQt6.QtCore import QObject, QSettings, Qt, pyqtSignal

from scripts.asmr_gui.controller import Controller
from scripts.asmr_gui import window as window_module
from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan
from scripts.asmr_transcribe_lib.gui_plan_io import save_plan
from scripts.asmr_transcribe_lib.gui_results import RunStore, atomic_json

SRT = "1\n00:00:00,000 --> 00:00:01,000\nSynthetic test.\n\n"


class QuietHealth(QObject):
    updated = pyqtSignal(dict)

    def __init__(self, **kwargs):
        super().__init__(kwargs.get("parent"))

    def configure(self, *args):
        pass

    def set_paused(self, *args):
        pass

    def check(self):
        return True

    def close(self):
        pass


class FakeScan:
    """Exercise the real controller's lifecycle executor with no native process."""
    def __init__(self, operation, state_root, command_factory, **kwargs):
        assert operation == "scan"
        self.run_id = str(uuid4())
        self.job = SimpleNamespace(name="deterministic-fake")
        self.run_dir = Path(state_root) / "operations" / self.run_id
        self.command_factory = command_factory

    def start(self):
        argv = self.command_factory(self)
        get = lambda key: argv[argv.index(key) + 1]
        plan = build_execution_plan(get("--root"), mode=get("--mode"),
            settings=json.loads(Path(get("--settings-file")).read_text(encoding="utf-8")),
            skip_existing_subtitles=True)
        digest = save_plan(plan, self.run_dir / "plan.json")
        atomic_json(self.run_dir / "scan.json", {"transport_hash": digest})

    def wait(self, **kwargs):
        return {"cleanup_verified": True, "result": {"complete": True}}

    def stop(self):
        return self.wait()


def library(path, names=("Work/Track01.wav",)):
    for name in names:
        audio = path / name
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"synthetic audio " + name.encode())
    return path


def inventory(path):
    return {p.relative_to(path).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def gui(qtbot, tmp_path, monkeypatch):
    preferences = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(window_module, "QSettings", lambda *args: preferences)
    owned = []

    def make(root=None):
        controller = Controller(state_root=tmp_path / "state", operation_factory=FakeScan,
                                health_monitor_factory=QuietHealth)
        if root:
            controller.set_root(str(root))
        window = window_module.BatchWindow(controller)
        qtbot.addWidget(window)
        owned.append((controller, window))
        return controller, window

    yield make, preferences
    for controller, window in owned:
        if controller.busy:
            controller.stop()
            qtbot.waitUntil(lambda: not controller.busy, timeout=5000)
        controller.request_close()
        window.close()


def test_startup_restores_input_root_and_scans(qtbot, tmp_path, gui):
    make, preferences = gui
    root = library(tmp_path / "previous")
    preferences.setValue("last_input_root", str(root))
    controller, window = make()
    with qtbot.waitSignal(controller.catalog_changed, timeout=5000,
                         check_params_cb=lambda: controller.state == "READY"):
        pass
    assert controller.root == str(root)
    assert window.root_edit.text() == str(root)
    assert window.tree_model.rowCount() == 1
    assert controller.run_model.summary()["planned"] == 1


def test_explicit_startup_root_takes_priority(qtbot, tmp_path, gui):
    make, preferences = gui
    preferences.setValue("last_input_root", str(library(tmp_path / "old")))
    root = library(tmp_path / "explicit", ("Other/Track09.wav",))
    controller, window = make(root)
    qtbot.waitUntil(lambda: controller.state == "READY", timeout=5000)
    assert controller.root == str(root)
    assert preferences.value("last_input_root") == str(root)


@pytest.mark.parametrize("action", ["browse", "edit"])
def test_new_directory_scans_without_old_statistics(qtbot, tmp_path, gui, monkeypatch, action):
    make, _ = gui
    first = library(tmp_path / "first")
    second = library(tmp_path / "second", ("Other/Track09.wav", "Other/Track10.wav"))
    controller, window = make(first)
    qtbot.waitUntil(lambda: controller.state == "READY", timeout=5000)
    old_ids = set(controller.run_model.tracks)
    if action == "browse":
        monkeypatch.setattr(window_module.QFileDialog, "getExistingDirectory", lambda *args: str(second))
        window.browse_button.click()
    else:
        window.root_edit.setText(str(second))
        window.root_edit.editingFinished.emit()
    assert controller.run_model.plan is None
    assert controller.run_model.summary()["planned"] == 0
    qtbot.waitUntil(lambda: controller.state == "READY", timeout=5000)
    assert controller.root == str(second)
    assert old_ids.isdisjoint(controller.run_model.tracks)
    summary = controller.run_model.summary()
    assert summary["planned"] == 2 and summary["success"] == summary["resolved"] == 0


def test_model_invariants_and_work_only_selection(qtbot, qtmodeltester, tmp_path, gui):
    make, _ = gui
    root = library(tmp_path / "library", ("Work/Track01.wav", "Work/Track02.wav"))
    controller, window = make(root)
    qtmodeltester.check(window.tree_model)
    qtmodeltester.check(window.proxy)
    qtbot.waitUntil(lambda: controller.state == "READY", timeout=5000)
    model = window.tree_model
    work = model.index(0, 0)
    track = model.index(0, 0, work)
    assert model.rowCount(work) == 2 and model.parent(track) == work
    assert model.flags(work) & Qt.ItemFlag.ItemIsUserCheckable
    assert not model.flags(track) & Qt.ItemFlag.ItemIsUserCheckable
    assert not model.setData(track, Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    assert model.setData(work, Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    assert not controller.run_model.selected_work_ids


def completed_history(root, state_root):
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    run_id = str(uuid4())
    directory = state_root / "runs" / run_id
    save_plan(plan, directory / "plan.json")
    store = RunStore(directory, run_id, plan, io.StringIO())
    store.start()

    def asr(*, transcribe_srt_path, **kwargs):
        transcribe_srt_path.write_text(SRT, encoding="utf-8")
        return transcribe_srt_path

    def copy(source, target):
        target.write_bytes(source.read_bytes())
        return target

    outcomes = run_plan(plan, project_root=root, transcriber=asr, translator=copy,
                        opencc_converter=copy, observer=store.observer)
    result = store.finish(outcomes)
    return directory, plan, result


def test_historical_display_is_read_only_without_resume_trust(qtbot, tmp_path, gui):
    make, _ = gui
    root = library(tmp_path / "library")
    directory, old_plan, result = completed_history(root, tmp_path / "state")
    before_audio, before_history = inventory(root), inventory(tmp_path / "state")
    controller, window = make(root)
    qtbot.waitUntil(lambda: controller.state in {"READY", "FINISHED", "ERROR"}, timeout=5000)
    assert controller.state == "FINISHED", controller.last_error
    assert controller.run_model.summary()["success"] == 1
    assert controller.last_run_dir == directory
    assert controller._trusted_runs == []
    assert controller._provenance(controller._new_intent("run")) == ()
    assert inventory(root) == before_audio
    for name, record in before_history.items():
        assert inventory(tmp_path / "state")[name] == record
    assert not (tmp_path / "state" / "gui-last-result.json").exists()


def test_current_session_resume_scan_remains_ready(qtbot, tmp_path, gui):
    make, _ = gui
    root = library(tmp_path / "library")
    directory, plan, result = completed_history(root, tmp_path / "state")
    controller, window = make(root)
    controller._trusted_runs.append((directory, result["run_id"], plan.plan_hash, str(root)))
    qtbot.waitUntil(lambda: not controller.busy and controller.run_model.plan is not None, timeout=5000)
    assert controller.state == "READY"


def test_history_outside_state_run_namespace_is_ignored(qtbot, tmp_path, gui):
    make, _ = gui
    root = library(tmp_path / "library")
    directory, _, result = completed_history(root, tmp_path / "foreign-state")
    atomic_json(tmp_path / "state/gui-last-result.json", {"schema": 1, "root": str(root),
                "directory": str(directory), "run_id": result["run_id"]})
    controller, window = make(root)
    qtbot.waitUntil(lambda: not controller.busy and controller.run_model.plan is not None, timeout=5000)
    assert controller.state == "READY" and controller._trusted_runs == []
    assert controller.run_model.summary()["success"] == 0


@pytest.mark.parametrize("corruption", ["audio", "touch", "add", "remove", "receipt", "result", "cross_root", "pointer_shape", "result_shape", "plan_shape", "nested_state", "nested_manifest", "nested_receipt"])
def test_stale_or_invalid_history_is_not_displayed(qtbot, tmp_path, gui, corruption):
    make, _ = gui
    root = library(tmp_path / "library")
    directory, _, _ = completed_history(root, tmp_path / "state")
    audio = root / "Work/Track01.wav"
    if corruption == "audio":
        audio.write_bytes(b"changed synthetic audio")
    elif corruption == "touch":
        import os
        stamp = audio.stat().st_mtime_ns
        os.utime(audio, ns=(stamp + 10000000, stamp + 10000000))
    elif corruption == "add":
        library(root, ("Work/Track02.wav",))
    elif corruption == "remove":
        audio.unlink()
    elif corruption == "receipt":
        next((directory / "receipts").glob("*.json")).unlink()
    elif corruption == "result":
        (directory / "result.json").write_text("{bad", encoding="utf-8")
    elif corruption == "pointer_shape":
        atomic_json(tmp_path / "state/gui-last-result.json", [])
        next((directory / "receipts").glob("*.json")).unlink()
    elif corruption == "result_shape":
        atomic_json(directory / "result.json", [])
    elif corruption == "plan_shape":
        atomic_json(directory / "plan.json", [])
    elif corruption in {"nested_state", "nested_manifest"}:
        path = directory / ("state.json" if corruption == "nested_state" else "plan_identity.json")
        record = json.loads(path.read_text(encoding="utf-8"))
        record["tracks"][next(iter(record["tracks"]))] = []
        atomic_json(path, record)
    elif corruption == "nested_receipt":
        path = next((directory / "receipts").glob("*.json"))
        record = json.loads(path.read_text(encoding="utf-8"))
        record["provenance"] = []
        atomic_json(path, record)
    else:
        root = library(tmp_path / "another")
        atomic_json(tmp_path / "state/gui-last-result.json", {"schema": 1, "root": str(root), "directory": str(directory)})
    controller, window = make(root)
    qtbot.waitUntil(lambda: controller.state in {"READY", "FINISHED", "ERROR"}, timeout=5000)
    assert controller.state == "READY", controller.last_error
    assert controller.run_model.summary()["success"] == 0
    assert not controller._trusted_runs


def test_user_directory_choice_supersedes_pending_startup(qtbot, tmp_path, gui, monkeypatch):
    make, preferences = gui
    previous = library(tmp_path / "previous")
    selected = library(tmp_path / "selected", ("Chosen/Track03.wav",))
    preferences.setValue("last_input_root", str(previous))
    controller, window = make()
    monkeypatch.setattr(window_module.QFileDialog, "getExistingDirectory", lambda *args: str(selected))
    window.browse_button.click()
    qtbot.waitUntil(lambda: not controller.busy, timeout=5000)
    assert controller.root == str(selected)
    assert controller.run_model.plan.root == selected


@pytest.mark.parametrize("candidate", ["empty", "missing", "file"])
def test_invalid_edited_root_cannot_start_previous_library(qtbot, tmp_path, gui, monkeypatch, candidate):
    make, _ = gui
    previous = library(tmp_path / "previous")
    controller, window = make(previous)
    qtbot.waitUntil(lambda: controller.state == "READY", timeout=5000)
    values = {"empty": "", "missing": str(tmp_path / "nonexistent"), "file": str(previous / "Work/Track01.wav")}
    window.root_edit.setText(values[candidate])
    calls = []
    monkeypatch.setattr(controller, "start", lambda: calls.append(controller.root))
    window.start_button.click()
    assert calls == []
    assert controller.state == "READY"
