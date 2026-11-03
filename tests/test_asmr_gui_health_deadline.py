"""Hard-deadline regression: partial headers must not extend availability probes."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_QT_API", "pyqt6")

from scripts.asmr_gui.health import probe_translation


def test_header_trickle_obeys_total_deadline(qapp):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(2)
    port = listener.getsockname()[1]
    stop = threading.Event()

    def serve():
        try:
            with listener.accept()[0] as peer:
                peer.recv(8192)
                for byte in b"HTTP/1.1 200 OK\r\nX-Slow: " + b"x" * 60:
                    if stop.wait(.025):
                        break
                    peer.sendall(bytes([byte]))
        except OSError:
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        result = probe_translation({"translate_api": f"http://127.0.0.1:{port}/v1", "translate_model": "test"}, .12)
        elapsed = time.monotonic() - started
        assert elapsed < .7, f"header trickle extended .12s deadline to {elapsed:.3f}s"
        assert result["error_code"] == "TRANSLATION_TIMEOUT"
    finally:
        stop.set()
        listener.close()
        thread.join(2)
        assert not thread.is_alive()


def test_monitor_close_does_not_hold_interpreter_exit(tmp_path):
    script = tmp_path / "close_monitor.py"
    script.write_text('''import sys, threading
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from PyQt6.QtCore import QCoreApplication, QTimer
from scripts.asmr_gui.health import HealthMonitor
app = QCoreApplication([])
entered = threading.Event()
never = threading.Event()
def stuck(*args):
    entered.set()
    never.wait()
monitor = HealthMonitor(project_root=Path.cwd(), asr_probe=stuck)
monitor.configure({})
timer = QTimer()
def close_when_entered():
    if entered.is_set():
        monitor.close()
        timer.stop()
        app.quit()
timer.timeout.connect(close_when_entered)
timer.start(5)
QTimer.singleShot(2000, app.quit)
app.exec()
assert entered.is_set()
''', encoding="utf-8")
    result = subprocess.run([sys.executable, "-B", str(script), str(Path(__file__).resolve().parents[1])],
                            capture_output=True, text=True, timeout=4)
    assert result.returncode == 0, result.stderr[-1000:]


def test_close_during_native_network_request_exits_cleanly(tmp_path):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(3)
    marker = tmp_path / "request-received"
    stop = threading.Event()

    def serve():
        try:
            with listener.accept()[0] as peer:
                peer.recv(8192)
                peer.sendall(b"HTTP/1.1 200 OK\r\nX-Pending: ")
                marker.write_text("request in flight", encoding="utf-8")
                stop.wait(3)
        except OSError:
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    script = tmp_path / "close_native_network.py"
    script.write_text('''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from PyQt6.QtCore import QCoreApplication, QTimer
from scripts.asmr_gui.health import HealthMonitor
app = QCoreApplication([])
monitor = HealthMonitor(project_root=Path.cwd(), asr_probe=lambda *args: {'available': True})
monitor.configure({'translate_api': sys.argv[2]})
timer = QTimer()
def close_when_received():
    if Path(sys.argv[3]).exists():
        monitor.close()
        timer.stop()
        app.quit()
timer.timeout.connect(close_when_received)
timer.start(5)
QTimer.singleShot(2500, app.quit)
app.exec()
assert Path(sys.argv[3]).exists()
''', encoding="utf-8")
    try:
        result = subprocess.run([sys.executable, "-B", str(script),
            str(Path(__file__).resolve().parents[1]), f"http://127.0.0.1:{listener.getsockname()[1]}/v1", str(marker)],
            capture_output=True, text=True, timeout=4)
        assert result.returncode == 0, result.stderr[-1000:]
    finally:
        stop.set()
        listener.close()
        thread.join(3)
        assert not thread.is_alive()
