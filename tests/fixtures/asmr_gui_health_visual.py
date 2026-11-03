"""Opt-in native, no-scan health evidence. Own-window captures; no user audio."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication
from scripts.asmr_gui.app_icon import application_icon, set_windows_app_id, WINDOWS_APP_ID
from scripts.asmr_gui.controller import Controller
from scripts.asmr_gui.health import HealthMonitor
from scripts.asmr_gui.theme import apply_theme
from scripts.asmr_gui import window as window_module
from asmr_gui_v1_visual import native_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("offline", "online", "real"), default="offline")
    args = parser.parse_args()
    base = Path(tempfile.gettempdir()) / "VoiceTransl" / "Tests" / ("health-native-" + uuid4().hex)
    base.mkdir(parents=True)
    (base / "owner.json").write_text(json.dumps({"kind": "VoiceTransl native health evidence", "path": str(base)}), encoding="utf-8")
    os.environ["QT_SCALE_FACTOR"] = "1"  # Native Windows desktop is 150%; do not multiply it.
    app_id_set = set_windows_app_id()
    app = QApplication([])
    app.setWindowIcon(application_icon())
    app.setQuitOnLastWindowClosed(False)
    apply_theme(app)
    server_state = {"online": args.profile == "online", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            server_state["requests"].append({"method": "GET", "path": self.path, "online": server_state["online"]})
            if not server_state["online"]:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            body = json.dumps({"data": [{"id": "controlled-sakura"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *unused):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.05), daemon=True)
    thread.start()
    controller = Controller(state_root=base / "state", project_root=PROJECT,
        health_monitor_factory=lambda **kwargs: HealthMonitor(**kwargs, interval_ms=1000, timeout=.5))
    if args.profile != "real":
        controller.apply_settings({"translate_api": f"http://127.0.0.1:{server.server_port}/v1", "translate_model": "controlled-sakura"})
    window_module.QSettings = lambda *unused: QSettings(str(base / "preferences.ini"), QSettings.Format.IniFormat)
    window = window_module.BatchWindow(controller)
    window.resize(1220, 840)
    window.show()
    window.raise_()
    evidence = {"profile": args.profile, "base": str(base), "app_id_set": app_id_set,
                "expected_app_id": WINDOWS_APP_ID, "captures": [], "transitions": []}
    ticks = []
    started = time.monotonic()
    step = 0
    failed = None

    def capture(name):
        window.repaint()
        pixmap = window.screen().grabWindow(int(window.winId()))
        if pixmap.isNull():
            raise RuntimeError("native own-window capture failed")
        path = base / (name + ".png")
        if not pixmap.save(str(path)):
            raise RuntimeError("capture save failed")
        evidence["captures"].append({"name": name, "path": str(path), "method": "QScreen.grabWindow of own visible Windows window",
            "geometry": native_geometry(window), "labels": [window.asr_label.text(), window.translation_label.text(), window.execution_label.text()],
            "tooltips": [window.asr_label.toolTip(), window.translation_label.toolTip()],
            "no_plan": controller.run_model.plan is None, "state": controller.state,
            "backend_button_enabled": window.backend_button.isEnabled(), "backend": controller.last_backend,
            "pixel_width": pixmap.width(), "pixel_height": pixmap.height()})

    def record(result):
        if not result.get("translation", {}).get("checking", True):
            evidence["transitions"].append({"elapsed": time.monotonic() - started, "result": result})

    controller.backend_changed.connect(record)
    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    heartbeat.start(10)
    timer = QTimer()

    def finish(error=None):
        nonlocal failed
        failed = error
        timer.stop()
        heartbeat.stop()
        evidence.update(error=error, elapsed=time.monotonic() - started, heartbeat_ticks=len(ticks),
            max_heartbeat_gap=max((b-a for a, b in zip(ticks, ticks[1:])), default=0), requests=server_state["requests"],
            cleanup=controller.request_close())
        (base / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        window.close()
        app.quit()

    def advance():
        nonlocal step
        try:
            if time.monotonic() - started > 20:
                finish("native evidence timeout")
                return
            stage = controller.last_backend.get("translation", {})
            if stage.get("checking", True):
                return
            if step == 0:
                if args.profile == "real" and not stage.get("available"):
                    finish("real Sakura not available")
                    return
                capture("fresh-" + args.profile)
                window.backend_button.click()
                assert controller.last_backend["translation"]["checking"]
                capture("manual-checking")
                step = 1
            elif step == 1:
                if args.profile == "real":
                    step = 3
                    return
                server_state["online"] = not server_state["online"]
                step = 2
            elif step == 2:
                if bool(stage.get("available")) != server_state["online"]:
                    return
                capture("automatic-" + ("online" if server_state["online"] else "offline"))
                if args.profile == "offline" and server_state["online"]:
                    server_state["online"] = False
                    return
                step = 3
            elif step == 3:
                window.toggle_maximized()
                step = 4
            elif step == 4:
                capture("maximized")
                assert window.isMaximized()
                window.toggle_maximized()
                step = 5
            elif step == 5:
                capture("restored")
                assert not window.isMaximized()
                # Read this process's own native icon handles; do not control other windows.
                user32 = ctypes.WinDLL("user32", use_last_error=True)
                user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
                user32.SendMessageW.restype = ctypes.c_ssize_t
                evidence["native_icon_handles"] = {str(kind): int(user32.SendMessageW(int(window.winId()), 0x007F, kind, 0)) for kind in (0, 1, 2)}
                evidence["qwindow_icon_sizes"] = [[size.width(), size.height()] for size in window.windowHandle().icon().availableSizes()]
                assert any(evidence["native_icon_handles"].values())
                finish()
        except Exception as exc:
            finish(type(exc).__name__ + ": " + str(exc))

    timer.timeout.connect(advance)
    timer.start(150)
    try:
        app.exec()
    finally:
        controller.request_close()
        server.shutdown()
        server.server_close()
        thread.join(1)
    print(json.dumps({"base": str(base), "error": failed, "captures": len(evidence["captures"])}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
