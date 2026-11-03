"""Bounded read-only availability monitoring, independent of execution preflight."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEventLoop, QObject, QTimer, Qt, QUrl, pyqtSignal
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkProxy, QNetworkReply, QNetworkRequest

from scripts.asmr_transcribe_lib.cli import _effective_transcribe_model
from scripts.asmr_transcribe_lib.gui_preflight import _local_model
from scripts.asmr_transcribe_lib.local_faster_whisper import (
    GUI_MODEL_PREFIX, find_faster_whisper_exe, find_faster_whisper_model_dir,
    resolve_faster_whisper_model_arg,
)
from .settings import default_settings

_ASR_KEYS = ("transcribe_backend", "transcribe_model", "kotoba_device", "transcribe_compute_type")


def probe_asr(settings, project_root):
    """Inspect local prerequisites only; the execution preflight validates CUDA."""
    backend = settings["transcribe_backend"]
    model = _effective_transcribe_model(backend, settings.get("transcribe_model"))
    device = settings["kotoba_device"] if backend == "kotoba" else "cuda"
    result = dict(required=True, available=False, backend=backend, device=device,
                  inference_proven=False, driver_checked=False, error_code=None)
    if backend == "xxl":
        executable = find_faster_whisper_exe(project_root)
        directory = find_faster_whisper_model_dir(project_root)
        local = (directory or Path(project_root) / "whisper-faster") / (GUI_MODEL_PREFIX + resolve_faster_whisper_model_arg(model))
        if executable is None or not executable.is_file():
            result["error_code"] = "ASR_EXECUTABLE_MISSING"
        elif not _local_model(local):
            result["error_code"] = "ASR_LOCAL_MODEL_MISSING"
    elif backend == "kotoba":
        local = Path(model)
        if not local.is_absolute() or not _local_model(local):
            result["error_code"] = "ASR_LOCAL_MODEL_REQUIRED"
        elif any(importlib.util.find_spec(name) is None for name in ("faster_whisper", "ctranslate2")):
            result["error_code"] = "ASR_DEPENDENCY_MISSING"
    else:
        result["error_code"] = "ASR_BACKEND_UNSUPPORTED"
    if not result["error_code"]:
        if device not in {"cpu", "cuda"}:
            result["error_code"] = "ASR_DEVICE_UNSUPPORTED"
        elif settings["transcribe_compute_type"] not in {"default", "auto", "int8", "int8_float16", "int8_float32", "int8_bfloat16", "int16", "float16", "float32", "bfloat16"}:
            result["error_code"] = "ASR_COMPUTE_TYPE_UNSUPPORTED"
        elif device == "cpu" and settings["transcribe_compute_type"] in {"float16", "int8_float16", "bfloat16", "int8_bfloat16"}:
            result["error_code"] = "ASR_DEVICE_COMPUTE_MISMATCH"
    result["available"] = result["error_code"] is None
    return result


def probe_translation(settings, timeout):
    """Only GET /models, with response-size and elapsed-time limits; no inference."""
    result = dict(required=True, available=False, api_reachable=False,
                  model_available=False, inference_proven=False, error_code=None)
    try:
        endpoint = urlsplit(settings["translate_api"])
        if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
                or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
            raise ValueError("invalid endpoint")
        endpoint.port  # Validate malformed ports without exporting URL/error text.
    except (ValueError, TypeError):
        result["error_code"] = "TRANSLATION_ENDPOINT_INVALID"
        return result
    if QCoreApplication.instance() is None:
        result["error_code"] = "TRANSLATION_CHECK_FAILED"
        return result
    # A wall-clock Qt timer aborts even trickling headers or pending DNS. All
    # network objects and their event loop belong to this probe thread.
    loop = QEventLoop()
    manager = QNetworkAccessManager(loop)
    manager.setProxy(QNetworkProxy(QNetworkProxy.ProxyType.NoProxy))
    request = QNetworkRequest(QUrl(settings["translate_api"].rstrip("/") + "/models"))
    request.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute,
                         QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
    reply = manager.get(request)
    reply.setReadBufferSize(65536)
    body = bytearray()
    timer = QTimer(loop)
    timer.setSingleShot(True)
    timer.setTimerType(Qt.TimerType.PreciseTimer)

    def expire():
        result["error_code"] = "TRANSLATION_TIMEOUT"
        reply.abort()
        loop.quit()

    def consume():
        body.extend(bytes(reply.read(262145 - len(body))))
        if len(body) > 262144:
            result["error_code"] = "TRANSLATION_RESPONSE_TOO_LARGE"
            reply.abort()
            loop.quit()

    reply.readyRead.connect(consume)
    reply.finished.connect(loop.quit)
    timer.timeout.connect(expire)
    timer.start(max(1, int(timeout * 1000)))
    try:
        loop.exec()
        timer.stop()
        status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        result["api_reachable"] = status is not None
        if result["error_code"]:
            return result
        if status is None:
            result["error_code"] = "TRANSLATION_CONNECTION_FAILED"
        elif status != 200:
            result["error_code"] = "TRANSLATION_HTTP_ERROR"
        elif reply.error() != QNetworkReply.NetworkError.NoError:
            result["error_code"] = "TRANSLATION_CONNECTION_FAILED"
        else:
            consume()
            if result["error_code"]:
                return result
            payload = json.loads(body)
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list):
                result["error_code"] = "TRANSLATION_RESPONSE_INVALID"
                return result
            result["model_available"] = any(isinstance(item, dict) and item.get("id") == settings["translate_model"] for item in data)
            result["available"] = result["model_available"]
            if not result["available"]:
                result["error_code"] = "TRANSLATION_MODEL_UNAVAILABLE"
    except (ValueError, UnicodeError):
        result["error_code"] = "TRANSLATION_RESPONSE_INVALID"
    finally:
        timer.stop()
        sip.delete(manager)
    return result


class HealthMonitor(QObject):
    """One in-flight probe; GUI-thread generations reject obsolete configurations."""
    updated = pyqtSignal(dict)
    _delivery = pyqtSignal(object)

    def __init__(self, *, project_root, interval_ms=15000, timeout=3,
                 asr_probe=None, translation_probe=None, parent=None):
        super().__init__(parent)
        self.project_root = Path(project_root)
        self.timeout = max(.05, float(timeout))
        self._asr_probe = asr_probe or probe_asr
        self._translation_probe = translation_probe or probe_translation
        self._generation = 0
        self._identity = None
        self._settings = {}
        self._asr_cache = None
        self._asr_identity = None
        self._inflight = self._pending = self._paused = self._closed = False
        self._last = {}
        self._delivery.connect(self._receive, Qt.ConnectionType.QueuedConnection)
        self._timer = QTimer(self)
        self._timer.setInterval(max(10, int(interval_ms)))
        self._timer.timeout.connect(self.check)

    def configure(self, settings, mode="full"):
        if self._closed:
            return
        values = default_settings()
        values.update(settings)
        asr_identity = tuple(values[key] for key in _ASR_KEYS)
        identity = (asr_identity, values["translate_api"], values["translate_model"], mode)
        if identity == self._identity:
            return
        self._generation += 1
        self._identity, self._settings, self._mode = identity, values, mode
        if asr_identity != self._asr_identity:
            self._asr_cache = None
        self._asr_identity = asr_identity
        self._last = {}
        self._timer.start()
        QTimer.singleShot(0, self.check)

    def set_paused(self, paused):
        was_paused = self._paused
        self._paused = bool(paused)
        if was_paused and not self._paused and not self._closed:
            QTimer.singleShot(0, self.check)

    def check(self):
        if self._closed or self._paused or self._identity is None:
            return False
        # Immediate visible feedback also applies when coalescing a manual check.
        checking = dict(self._last)
        checking["asr"] = dict(self._asr_cache or {}, checking=self._asr_cache is None)
        checking["translation"] = dict(checking.get("translation", {}), checking=self._mode == "full",
                                       required=self._mode == "full")
        self.updated.emit(checking)
        if self._inflight:
            self._pending = True
            return True
        self._inflight = True
        self._pending = False
        generation = self._generation
        settings = dict(self._settings)
        mode = self._mode
        cached_asr = self._asr_cache
        # One in-flight, daemon probe cannot keep Python alive after GUI close
        # if a local filesystem or driver operation stalls outside our control.
        threading.Thread(target=self._probe, args=(generation, settings, mode, cached_asr),
                         name="asmr-health", daemon=True).start()
        return True

    def _probe(self, generation, settings, mode, cached_asr):
        try:
            asr = cached_asr or self._asr_probe(settings, self.project_root)
        except Exception:
            asr = dict(available=False, error_code="ASR_CHECK_FAILED", inference_proven=False)
        try:
            translation = self._translation_probe(settings, self.timeout) if mode == "full" else dict(
                required=False, available=False, api_reachable=False, model_available=False,
                inference_proven=False, error_code=None)
        except Exception:
            translation = dict(available=False, api_reachable=False, model_available=False,
                               inference_proven=False, error_code="TRANSLATION_CHECK_FAILED")
        result = dict(source="health", asr=dict(asr, checking=False),
                      translation=dict(translation, checking=False), inference_proven=False)
        if not self._closed:
            self._delivery.emit((generation, result))

    def _receive(self, message):
        if self._closed:
            return
        generation, result = message
        self._inflight = False
        if generation == self._generation:
            self._asr_cache = result["asr"]
            self._last = result
            self.updated.emit(result)
        else:
            self._pending = True
        if self._pending and not self._paused:
            self._pending = False
            QTimer.singleShot(0, self.check)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        self._pending = False
        self._timer.stop()
