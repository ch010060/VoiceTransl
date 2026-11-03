"""Native GUI evidence host using synthetic audio and injected worker adapters.

Its trusted local JSON control file is a test fixture; production entrypoints
expose no simulation flags. It can capture its own shown window and records the
capture method explicitly. It never captures or controls another application.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).with_name("asmr_gui_v1_driver.py")
sys.path.insert(0, str(PROJECT))

from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication
from scripts.asmr_gui.controller import Controller
from scripts.asmr_gui import window as window_module
from scripts.asmr_gui.theme import apply_theme
from scripts.asmr_gui_runtime import ManagedOperation, ManagedWorker
from scripts.asmr_transcribe_lib.gui_plan_io import encode_plan


def synthetic_library(root):
    root = Path(root).resolve()
    marker = root / ".synthetic-gui-fixture"
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError("refusing to use a non-fixture library")
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text("Synthetic GUI evidence only; no real audio or inference.\n", encoding="utf-8")
    names = (
        "RJ100001 雨の日の朗読/01 はじまり.wav",
        "RJ100001 雨の日の朗読/情景/02 静かな午後.flac",
        "RJ100001 雨の日の朗読/a/voice03.flac",
        "RJ100001 雨の日の朗読/b/voice03 SEなし.wav",
        "RJ100002 失敗時の表示確認/fail01.wav",
        "RJ100003 ゆっくり進む合成テスト/slow01.wav",
        "RJ100003 ゆっくり進む合成テスト/情景/slow02.wav",
        "RJ100004 長い日本語の作品タイトルと階層表示を確認するためのサンプル/01 おだやかな案内.wav",
        "_root/01 実在するルート名.wav",
        "00 合成ルート直下.wav",
    )
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"SYNTHETIC_GUI_AUDIO_NO_INFERENCE")
    (root / "RJ100005 音軌のない作品").mkdir(exist_ok=True)
    return root


def operation_factory(operation, state_root, command_factory, **kwargs):
    def command(managed):
        return [str(DRIVER) if str(arg).replace("\\", "/").endswith("scripts/asmr_gui_worker.py") else arg
                for arg in command_factory(managed)]
    return ManagedOperation(operation, state_root, command, **kwargs)


def worker_command(managed):
    return [sys.executable, "-B", "-u", str(DRIVER), "run", "--run-id", managed.run_id,
        "--job-name", managed.job.name, "--state-root", str(managed.state_root),
        "--project-root", str(PROJECT), "--plan", str(managed.run_dir / "plan.json"),
        "--expected-digest", encode_plan(managed.plan)["transport_hash"]]


def worker_factory(*args, **kwargs):
    # Keep STOPPING visible long enough for a native screenshot, while preserving
    # the same actual Job cleanup and ownership checks as production.
    return ManagedWorker(*args, cooperative_timeout=3, cleanup_timeout=5, **kwargs)


def rect_values(rect):
    return {"x": rect.x(), "y": rect.y(), "width": rect.width(), "height": rect.height()}


def native_geometry(window):
    result = {"qt_geometry": rect_values(window.geometry()),
        "qt_frame_geometry": rect_values(window.frameGeometry()),
        "qt_maximized": window.isMaximized(), "qt_minimized": window.isMinimized(),
        "device_pixel_ratio": window.devicePixelRatioF(),
        "qt_platform": QApplication.platformName(),
        "requested_qt_scale_factor": os.environ.get("QT_SCALE_FACTOR", "system default")}
    screen = window.screen()
    if screen:
        result["screen"] = {"name": screen.name(), "geometry": rect_values(screen.geometry()),
            "available_geometry": rect_values(screen.availableGeometry()),
            "logical_dpi": screen.logicalDotsPerInch(), "physical_dpi": screen.physicalDotsPerInch()}
    if os.name == "nt":
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        hwnd = wintypes.HWND(int(window.winId()))
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL
        user32.IsZoomed.argtypes = [wintypes.HWND]
        user32.IsZoomed.restype = wintypes.BOOL
        native = wintypes.RECT()
        result["hwnd"] = int(window.winId())
        if user32.GetWindowRect(hwnd, ctypes.byref(native)):
            result["native_rect"] = {"left": native.left, "top": native.top,
                "right": native.right, "bottom": native.bottom,
                "width": native.right - native.left, "height": native.bottom - native.top}
        result["native_zoomed"] = bool(user32.IsZoomed(hwnd))
        if hasattr(user32, "GetDpiForWindow"):
            user32.GetDpiForWindow.argtypes = [wintypes.HWND]
            user32.GetDpiForWindow.restype = wintypes.UINT
            result["native_dpi"] = int(user32.GetDpiForWindow(hwnd))
    return result


class EvidenceHost:
    def __init__(self, args):
        self.args = args
        self.evidence = Path(args.evidence_dir).resolve()
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.root = synthetic_library(args.library)
        self.control = Path(args.control_file).resolve()
        self.last_request = None
        self.command_result = None
        self.last_capture = None
        self.started = time.monotonic()
        self.notification_count = 0
        self.states = []
        self.controller = Controller(state_root=Path(args.state_root).resolve(), project_root=PROJECT,
            python_executable=sys.executable, operation_factory=operation_factory,
            worker_factory=worker_factory, worker_command=worker_command)
        # Test fixture preferences are kept beside evidence, never user settings.
        window_module.QSettings = lambda *unused: QSettings(str(self.evidence / "preferences.ini"), QSettings.Format.IniFormat)
        self.window = window_module.BatchWindow(self.controller)
        self.window.setWindowTitle("VoiceTransl ASMR GUI — Synthetic evidence fixture")
        self.controller.state_changed.connect(self.record_state)
        self.controller.completed.connect(self.completed)
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.snapshot_timer = QTimer()
        self.snapshot_timer.setInterval(500)
        self.snapshot_timer.timeout.connect(self.snapshot)
        self.controller.set_root(str(self.root))
        self.window.root_edit.setText(str(self.root))
        self.window.show()
        self.window.raise_()
        self.controller.log_message.emit("SYNTHETIC GUI fixture · no real audio/model inference")
        self.controller.scan()
        self.timer.start()
        self.snapshot_timer.start()

    def completed(self, result):
        self.notification_count += 1
        self.snapshot()

    def record_state(self, state):
        self.states.append({"state": state, "elapsed_s": time.monotonic() - self.started})
        self.snapshot()

    def snapshot(self):
        geometry = native_geometry(self.window)
        essential = {}
        for name in ("start_button", "stop_button", "scan_button", "settings_button"):
            button = getattr(self.window, name)
            bounds = button.rect()
            essential[name] = {"visible": button.isVisible(), "enabled": button.isEnabled(),
                "top_left_inside": self.window.rect().contains(button.mapTo(self.window, bounds.topLeft())),
                "bottom_right_inside": self.window.rect().contains(button.mapTo(self.window, bounds.bottomRight()))}
        plan = self.controller.run_model.plan
        document = {"synthetic": True, "inference_proven": False, "pid": os.getpid(),
            "state": self.controller.state, "last_error": self.controller.last_error,
            "states": self.states, "geometry": geometry, "controls": essential,
            "summary": self.controller.run_model.summary(), "notifications": self.notification_count,
            "last_run_dir": str(self.controller.last_run_dir or ""),
            "last_cleanup": self.controller.last_cleanup, "command_result": self.command_result,
            "capture": self.last_capture,
            "works": [{"name": w.relative_path or "@root", "kind": w.kind, "id": w.work_id,
                "selected": w.work_id in self.controller.run_model.selected_work_ids} for w in plan.works] if plan else [],
            "tracks": [{"id": tid, "status": state.status, "stage": state.stage, "ratio": state.ratio,
                "warnings": state.warnings} for tid, state in self.controller.run_model.tracks.items()]}
        temporary = self.evidence / "latest.tmp"
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            temporary.replace(self.evidence / "latest.json")
        except PermissionError:
            # A concurrent evidence reader may briefly hold the Windows file;
            # the next bounded timer tick publishes the same current state.
            pass
        return document

    def poll(self):
        if not self.control.is_file():
            return
        try:
            command = json.loads(self.control.read_text(encoding="utf-8-sig"))
            request = command.get("request_id")
            if request is None or request == self.last_request:
                return
            self.last_request = request
            action = command["action"]
            result = self.perform(action, command)
            self.command_result = {"request_id": request, "action": action, "result": result}
            document = self.snapshot()
            (self.evidence / (str(request).replace("/", "_").replace("\\", "_") + ".json")).write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self.command_result = {"error": type(exc).__name__}

    def perform(self, action, command):
        if action == "scan":
            return self.controller.scan()
        if action == "start":
            return self.controller.start()
        if action == "stop":
            return self.controller.stop()
        if action == "select":
            if not self.controller.safe_to_edit or self.controller.run_model.plan is None:
                return False
            names = command["works"]
            for work in self.controller.run_model.plan.works:
                name = work.relative_path or "@root"
                selected = any(token in name for token in names)
                self.controller.run_model.set_work_selected(work.work_id, selected)
            return sorted(self.controller.run_model.selected_work_ids)
        if action == "resize":
            self.window.showNormal()
            self.window.resize(int(command["width"]), int(command["height"]))
            return True
        if action == "maximize":
            self.window.showMaximized()
            return True
        if action == "restore":
            self.window.showNormal()
            return True
        if action == "logs":
            self.window.log_toggle.setChecked(bool(command["visible"]))
            self.window._toggle_logs()
            return True
        if action == "filter":
            index = self.window.filter_combo.findData(command.get("status", "all"))
            if index < 0:
                raise ValueError("unknown visual filter")
            self.window.filter_combo.setCurrentIndex(index)
            return True
        if action == "settings":
            return self.controller.apply_settings(command.get("settings", {}),
                mode=command.get("mode", "full"), limit=command.get("limit"))
        if action == "snapshot":
            return True
        if action == "capture":
            return self.capture(command)
        if action == "layout_audit":
            return self.layout_audit()
        if action == "quit":
            self.window.close()
            return True
        raise ValueError("unknown fixture action")

    def capture(self, command):
        """Capture this fixture's own shown window only, with explicit method."""
        if not self.window.isVisible() or self.window.isMinimized():
            raise ValueError("own evidence window must be visible and restored")
        if command.get("expand"):
            self.window.tree.expandAll()
        path = Path(command.get("path", self.evidence / "own-window.png")).resolve()
        if not path.is_relative_to(self.evidence) or path.suffix.lower() != ".png":
            raise ValueError("capture output must be PNG inside evidence directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.window.repaint()
        QApplication.processEvents()
        pixmap = self.window.screen().grabWindow(int(self.window.winId()))
        method = "QScreen.grabWindow of own visible window"
        colors = self.sample_colors(pixmap)
        if pixmap.isNull() or colors < 8:
            pixmap = self.window.grab()
            method = "QWidget.grab of own shown window; rendered widget capture"
            colors = self.sample_colors(pixmap)
        if pixmap.isNull() or colors < 8 or not pixmap.save(str(path), "PNG"):
            raise RuntimeError("own-window capture was blank or could not be saved")
        self.last_capture = {"path": str(path), "method": method, "sample_colors": colors,
            "pixel_width": pixmap.width(), "pixel_height": pixmap.height(),
            "pixmap_device_pixel_ratio": pixmap.devicePixelRatio(),
            "native_visible": QApplication.platformName() == "windows" and self.window.isVisible(),
            "geometry": native_geometry(self.window), "state": self.controller.state,
            "synthetic": True, "inference_proven": False}
        return self.last_capture

    @staticmethod
    def sample_colors(pixmap):
        if pixmap.isNull():
            return 0
        image = pixmap.toImage()
        points = ((x, y) for x in range(0, image.width(), max(1, image.width() // 32))
                  for y in range(0, image.height(), max(1, image.height() // 32)))
        return len({image.pixel(x, y) for x, y in points})

    def layout_audit(self):
        original = self.window.size()
        original_maximized = self.window.isMaximized()
        observations = []
        self.window.showNormal()
        try:
            for width, height in ((980, 600), (1220, 840), (1480, 1000)):
                self.window.resize(width, height)
                QApplication.processEvents()
                document = self.snapshot()
                observations.append({"requested": {"width": width, "height": height},
                    "geometry": document["geometry"], "controls": document["controls"],
                    "tree_height": self.window.tree.height(),
                    "all_essential_inside": all(c["visible"] and c["top_left_inside"] and c["bottom_right_inside"]
                                                for c in document["controls"].values())})
        finally:
            self.window.resize(original)
            if original_maximized:
                self.window.showMaximized()
        path = self.evidence / "layout-audit.json"
        path.write_text(json.dumps({"synthetic": True, "observations": observations},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return {"path": str(path), "all_essential_inside": all(o["all_essential_inside"] for o in observations),
            "observations": observations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("state-root", "library", "control-file", "evidence-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--scale-factor", type=float,
        help="Test-only Qt scale multiplier between 0.5 and 2; does not change Windows DPI")
    args = parser.parse_args()
    if args.scale_factor is not None:
        if not .5 <= args.scale_factor <= 2:
            parser.error("--scale-factor must be between 0.5 and 2")
        os.environ["QT_SCALE_FACTOR"] = str(args.scale_factor)
    app = QApplication([])
    apply_theme(app)
    host = EvidenceHost(args)
    app.aboutToQuit.connect(host.snapshot)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
