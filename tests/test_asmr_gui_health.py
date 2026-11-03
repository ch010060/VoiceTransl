"""Bounded, read-only backend monitoring without models or external services."""
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication

from scripts.asmr_gui.health import HealthMonitor

@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    instance.setQuitOnLastWindowClosed(False)
    return instance


def wait_for(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    assert predicate(), "bounded health condition timed out"


def ready_asr(settings, project_root):
    return {"available": True, "backend": settings["transcribe_backend"],
            "device": "cpu", "error_code": None, "inference_proven": False}


def ready_translation(settings, timeout):
    return {"available": True, "api_reachable": True, "model_available": True,
            "error_code": None, "inference_proven": False}


def complete(spy):
    return [row[0] for row in spy if row[0].get("translation")
            and not row[0]["translation"].get("checking")]


@pytest.fixture
def monitors(app, tmp_path):
    owned = []

    def make(**kwargs):
        kwargs.setdefault("asr_probe", ready_asr)
        kwargs.setdefault("translation_probe", ready_translation)
        kwargs.setdefault("interval_ms", 100000)
        monitor = HealthMonitor(project_root=tmp_path, **kwargs)
        owned.append(monitor)
        return monitor

    yield make
    for monitor in owned:
        monitor.close()
    app.processEvents()


@pytest.fixture
def server():
    state = {"mode": "online", "models": ["sakura-exact"], "requests": [],
             "delay": 0, "secret": "PRIVATE_TOKEN_RESPONSE_123"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(("GET", self.path))
            mode = state["mode"]
            if mode == "disconnect":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            time.sleep(state["delay"])
            if mode == "error":
                body, status = state["secret"].encode(), 503
            elif mode == "malformed":
                body, status = json.dumps({"private": state["secret"]}).encode(), 200
            else:
                body = json.dumps({"data": [{"id": name} for name in state["models"]]}).encode()
                status = 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self):
            state["requests"].append(("POST", self.path))
            self.send_error(405)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    worker = threading.Thread(target=partial(httpd.serve_forever, poll_interval=.02), daemon=True)
    worker.start()
    state["url"] = f"http://127.0.0.1:{httpd.server_port}/v1"
    yield state
    httpd.shutdown()
    httpd.server_close()
    worker.join(timeout=1)
    assert not worker.is_alive()


def configure_http(monitor, server, **settings):
    monitor.configure({"translate_api": server["url"], "translate_model": "sakura-exact", **settings})


def test_startup_and_manual_health_need_no_scan_or_selection(app, tmp_path):
    from scripts.asmr_gui.controller import Controller
    created = []

    def factory(**kwargs):
        monitor = HealthMonitor(**kwargs, interval_ms=100000,
                                asr_probe=ready_asr, translation_probe=ready_translation)
        created.append(monitor)
        return monitor

    controller = Controller(state_root=tmp_path / "state", project_root=tmp_path,
                            health_monitor_factory=factory)
    spy = QSignalSpy(controller.backend_changed)
    try:
        assert controller.root == "" and controller.run_model.plan is None
        wait_for(app, lambda: bool(complete(spy)))
        assert controller.last_backend["translation"]["available"]
        assert controller.state == "IDLE"
        before = len(complete(spy))
        assert controller.check_backends()
        assert spy[-1][0]["translation"]["checking"]
        wait_for(app, lambda: len(complete(spy)) > before)
        assert controller.state == "IDLE" and controller.run_model.plan is None
        saved = controller.last_backend
        assert controller.set_root(str(tmp_path / "unscanned"))
        assert controller.last_backend == saved
    finally:
        assert controller.request_close()


def test_local_http_exact_model_and_no_inference(app, monitors, server):
    monitor = monitors(translation_probe=None)
    spy = QSignalSpy(monitor.updated)
    configure_http(monitor, server)
    wait_for(app, lambda: bool(complete(spy)))
    stage = complete(spy)[-1]["translation"]
    assert stage["available"] and stage["api_reachable"] and stage["model_available"]
    assert not stage["inference_proven"] and not complete(spy)[-1]["inference_proven"]
    server["models"] = ["sakura-exact-suffix"]
    before = len(complete(spy))
    assert monitor.check()
    wait_for(app, lambda: len(complete(spy)) > before)
    stage = complete(spy)[-1]["translation"]
    assert stage["api_reachable"] and not stage["model_available"] and not stage["available"]
    assert stage["error_code"]
    assert server["requests"] and all(row == ("GET", "/v1/models") for row in server["requests"])


def test_automatic_polling_recovers_offline_online_offline(app, monitors, server):
    server["mode"] = "disconnect"
    monitor = monitors(interval_ms=35, translation_probe=None, timeout=.1)
    spy = QSignalSpy(monitor.updated)
    configure_http(monitor, server)
    wait_for(app, lambda: bool(complete(spy)))
    assert not complete(spy)[-1]["translation"]["available"]
    assert not complete(spy)[-1]["translation"]["api_reachable"]
    server["mode"] = "online"
    wait_for(app, lambda: any(row["translation"]["available"] for row in complete(spy)))
    before = len(complete(spy))
    server["mode"] = "disconnect"
    wait_for(app, lambda: len(complete(spy)) > before and
             not complete(spy)[-1]["translation"]["available"])
    assert not complete(spy)[-1]["translation"]["api_reachable"]
    assert all(method == "GET" for method, _ in server["requests"])


def test_http_timeout_keeps_gui_heartbeat_responsive(app, monitors, server):
    server["delay"] = .35
    monitor = monitors(translation_probe=None, timeout=.05)
    spy = QSignalSpy(monitor.updated)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start(5)
    try:
        started = time.monotonic()
        configure_http(monitor, server)
        assert time.monotonic() - started < .2
        wait_for(app, lambda: bool(spy))
        assert any(row[0]["translation"]["checking"] for row in spy)
        wait_for(app, lambda: bool(complete(spy)), timeout=3)
        assert time.monotonic() - started < 3
        assert len(ticks) >= 3
        assert not complete(spy)[-1]["translation"]["available"]
        assert complete(spy)[-1]["translation"]["error_code"]
    finally:
        timer.stop()


@pytest.mark.parametrize("mode", ["error", "malformed"])
def test_http_errors_do_not_export_response_text(app, monitors, server, mode):
    server["mode"] = mode
    monitor = monitors(translation_probe=None)
    spy = QSignalSpy(monitor.updated)
    configure_http(monitor, server)
    wait_for(app, lambda: bool(complete(spy)))
    result = complete(spy)[-1]
    assert not result["translation"]["available"]
    assert result["translation"]["error_code"]
    assert server["secret"] not in json.dumps(list(spy))
    assert all(not stage["inference_proven"] for stage in (result["asr"], result["translation"]))


def test_probe_errors_are_redacted_and_do_not_claim_inference(app, monitors):
    def broken(*args):
        raise RuntimeError("private-path-and-token-SECRET_456")

    monitor = monitors(asr_probe=broken, translation_probe=broken)
    spy = QSignalSpy(monitor.updated)
    monitor.configure({})
    wait_for(app, lambda: bool(complete(spy)))
    result = complete(spy)[-1]
    for name in ("asr", "translation"):
        assert not result[name]["available"] and result[name]["error_code"]
        assert not result[name]["inference_proven"]
    assert "SECRET_456" not in json.dumps(list(spy))


def test_stale_configuration_results_are_rejected(app, monitors):
    entered, release = threading.Event(), threading.Event()
    seen = []

    def probe(settings, timeout):
        name = settings["translate_model"]
        seen.append(name)
        if name == "old":
            entered.set()
            release.wait(3)
        return dict(ready_translation(settings, timeout), available=name == "new")

    monitor = monitors(translation_probe=probe)
    spy = QSignalSpy(monitor.updated)
    try:
        monitor.configure({"translate_model": "old"})
        wait_for(app, entered.is_set)
        monitor.configure({"translate_model": "new"})
        boundary = len(spy)
        release.set()
        wait_for(app, lambda: bool(complete(spy)))
        assert "new" in seen
        assert complete(spy)[-1]["translation"]["available"]
        assert not any(not row[0]["translation"].get("checking") and
                       not row[0]["translation"]["available"] for row in list(spy)[boundary:])
    finally:
        release.set()


def test_manual_and_timer_requests_do_not_overlap_probes(app, monitors):
    entered, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    active, maximum, calls = 0, 0, 0

    def probe(settings, timeout):
        nonlocal active, maximum, calls
        with lock:
            calls += 1
            active += 1
            maximum = max(maximum, active)
        entered.set()
        release.wait(3)
        with lock:
            active -= 1
        return ready_translation(settings, timeout)

    monitor = monitors(translation_probe=probe, interval_ms=15)
    spy = QSignalSpy(monitor.updated)
    try:
        monitor.configure({})
        wait_for(app, entered.is_set)
        for _ in range(10):
            assert monitor.check()
        QTest.qWait(80)
        assert calls == 1 and maximum == 1
        release.set()
        wait_for(app, lambda: bool(complete(spy)))
        assert maximum == 1
    finally:
        release.set()


def test_close_is_nonblocking_and_ignores_late_results(app, monitors):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def probe(settings, timeout):
        entered.set()
        release.wait(3)
        finished.set()
        return ready_translation(settings, timeout)

    monitor = monitors(translation_probe=probe)
    spy = QSignalSpy(monitor.updated)
    try:
        monitor.configure({})
        wait_for(app, entered.is_set)
        started = time.monotonic()
        monitor.close()
        assert time.monotonic() - started < .2
        boundary = len(spy)
        assert not monitor.check()
        release.set()
        wait_for(app, finished.is_set)
        QTest.qWait(50)
        assert len(spy) == boundary
        monitor.close()
    finally:
        release.set()


def test_asr_is_cached_until_relevant_settings_change(app, monitors):
    calls = []

    def asr(settings, project_root):
        calls.append(dict(settings))
        return ready_asr(settings, project_root)

    monitor = monitors(asr_probe=asr)
    spy = QSignalSpy(monitor.updated)
    monitor.configure({})
    wait_for(app, lambda: bool(complete(spy)))
    assert len(calls) == 1
    monitor.configure({"parallel": 3})
    QTest.qWait(25)
    assert len(calls) == 1
    before = len(complete(spy))
    monitor.configure({"translate_model": "different-sakura"})
    wait_for(app, lambda: len(complete(spy)) > before)
    assert len(calls) == 1
    before = len(complete(spy))
    assert monitor.check()
    wait_for(app, lambda: len(complete(spy)) > before)
    assert len(calls) == 1
    before = len(complete(spy))
    monitor.configure({"transcribe_model": "different-local-model"})
    wait_for(app, lambda: len(complete(spy)) > before)
    assert len(calls) == 2


def test_paused_monitor_does_not_probe_until_resumed(app, monitors):
    calls = []
    asr_calls = []

    def asr(settings, project_root):
        asr_calls.append(settings["transcribe_model"])
        return ready_asr(settings, project_root)

    def probe(settings, timeout):
        calls.append(settings["translate_model"])
        return ready_translation(settings, timeout)

    monitor = monitors(asr_probe=asr, translation_probe=probe, interval_ms=15)
    spy = QSignalSpy(monitor.updated)
    monitor.set_paused(True)
    monitor.configure({})
    assert not monitor.check()
    QTest.qWait(70)
    assert calls == [] and asr_calls == []
    monitor.set_paused(False)
    wait_for(app, lambda: bool(complete(spy)))
    monitor.set_paused(True)
    before = len(calls)
    before_asr = len(asr_calls)
    monitor.configure({"transcribe_model": "changed-while-running"})
    QTest.qWait(70)
    assert not monitor.check() and len(calls) == before and len(asr_calls) == before_asr
    monitor.set_paused(False)
    wait_for(app, lambda: len(asr_calls) > before_asr)


def test_transcribe_only_never_contacts_translation(app, monitors):
    calls = []

    def forbidden(settings, timeout):
        calls.append(settings)
        raise AssertionError("transcribe-only must not contact Sakura")

    monitor = monitors(translation_probe=forbidden, interval_ms=15)
    spy = QSignalSpy(monitor.updated)
    monitor.configure({}, mode="transcribe_only")
    wait_for(app, lambda: bool(complete(spy)))
    QTest.qWait(50)
    assert calls == []
    assert complete(spy)[-1]["translation"]["required"] is False
    assert complete(spy)[-1]["translation"]["inference_proven"] is False


@pytest.mark.parametrize("endpoint", [
    "http://user:PRIVATE_TOKEN@127.0.0.1:1/v1", "http://127.0.0.1:1/v1?token=PRIVATE_TOKEN",
    "file:///private/path/PRIVATE_TOKEN", "http://127.0.0.1:invalid/PRIVATE_TOKEN",
])
def test_invalid_endpoints_are_rejected_without_leaking_details(app, monitors, endpoint):
    monitor = monitors(translation_probe=None)
    spy = QSignalSpy(monitor.updated)
    monitor.configure({"translate_api": endpoint})
    wait_for(app, lambda: bool(complete(spy)))
    stage = complete(spy)[-1]["translation"]
    assert stage["error_code"] == "TRANSLATION_ENDPOINT_INVALID"
    assert not stage["api_reachable"] and not stage["model_available"]
    assert "PRIVATE_TOKEN" not in json.dumps(list(spy))


def test_health_executor_cannot_starve_controller_lifecycle(app, tmp_path):
    from scripts.asmr_gui.controller import Controller
    entered, release = threading.Event(), threading.Event()

    def probe(settings, timeout):
        entered.set()
        release.wait(3)
        return ready_translation(settings, timeout)

    def factory(**kwargs):
        return HealthMonitor(**kwargs, interval_ms=100000, asr_probe=ready_asr,
                             translation_probe=probe)

    controller = Controller(state_root=tmp_path / "state", project_root=tmp_path,
                            health_monitor_factory=factory)
    try:
        wait_for(app, entered.is_set)
        future = controller._executor.submit(lambda: "lifecycle-remains-available")
        wait_for(app, future.done, timeout=1)
        assert future.result() == "lifecycle-remains-available"
        assert controller.state == "IDLE"
    finally:
        release.set()
        assert controller.request_close()


def test_running_controller_pauses_resource_probes_without_lifecycle_change(app, tmp_path):
    from scripts.asmr_gui.controller import Controller
    asr_calls, translation_calls = [], []

    def asr(settings, project_root):
        asr_calls.append(True)
        return ready_asr(settings, project_root)

    def translation(settings, timeout):
        translation_calls.append(True)
        return ready_translation(settings, timeout)

    def factory(**kwargs):
        return HealthMonitor(**kwargs, interval_ms=15, asr_probe=asr,
                             translation_probe=translation)

    controller = Controller(state_root=tmp_path / "state", project_root=tmp_path,
                            health_monitor_factory=factory)
    spy = QSignalSpy(controller.backend_changed)
    try:
        wait_for(app, lambda: bool(complete(spy)))
        controller._set_state("RUNNING")
        counts = len(asr_calls), len(translation_calls)
        assert not controller.check_backends()
        QTest.qWait(80)
        assert (len(asr_calls), len(translation_calls)) == counts
        assert controller.state == "RUNNING"
        controller._set_state("IDLE")
        wait_for(app, lambda: len(translation_calls) > counts[1])
        assert len(asr_calls) == counts[0]
    finally:
        controller._set_state("IDLE")
        assert controller.request_close()
