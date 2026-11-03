"""Read-only availability checks. No model construction, download or inference."""
from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
from urllib.parse import urlsplit

from .local_faster_whisper import (
    GUI_MODEL_PREFIX, find_faster_whisper_exe, find_faster_whisper_model_dir,
    resolve_faster_whisper_model_arg,
)
from .openai_compatible import preflight_model


def cuda_available() -> bool:
    """Probe the installed driver only; never load a transcription model."""
    if os.name != "nt":
        return False
    try:
        driver = ctypes.WinDLL("nvcuda.dll")
        count = ctypes.c_int()
        return driver.cuInit(0) == 0 and driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
    except (OSError, AttributeError):
        return False


def _local_model(path):
    return path.is_dir() and all((path / name).is_file() and (path / name).stat().st_size > 0
                                 for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"))


def preflight(plan, project_root, client=None, *, runtime_probe=None) -> dict:
    plan.assert_ready()
    settings = plan.settings
    processes = [track for track in plan.planned_tracks if track.task.action == "process"]
    asr_needed = any(not track.task.transcribe_srt_path.exists() for track in processes)
    # The existing pipeline regenerates translation even when reusing ASR.
    translation_needed = plan.mode == "full" and bool(processes)
    conversion_needed = plan.mode == "full" and bool(processes)
    asr = {"required": asr_needed, "available": not asr_needed, "backend": settings["transcribe_backend"],
           "device": settings["kotoba_device"] if settings["transcribe_backend"] == "kotoba" else "cuda",
           "inference_proven": False, "error_code": None}
    translation = {"required": translation_needed, "available": not translation_needed,
                   "inference_proven": False, "error_code": None}
    conversion = {"required": conversion_needed, "available": not conversion_needed, "error_code": None}
    # The existing pipeline imports its default converter even in ASR-only
    # mode. Report that runtime prerequisite before importing execution code.
    runtime = {"required": True, "available": importlib.util.find_spec("opencc") is not None,
               "error_code": None}
    if not runtime["available"]:
        runtime["error_code"] = "PIPELINE_DEPENDENCY_MISSING"
    if asr_needed:
        backend = settings["transcribe_backend"]
        model = settings["transcribe_model"]
        if backend == "xxl":
            executable = find_faster_whisper_exe(project_root)
            directory = find_faster_whisper_model_dir(project_root)
            name = GUI_MODEL_PREFIX + resolve_faster_whisper_model_arg(model)
            local = directory / name if directory else Path(project_root) / "whisper-faster" / name
            if executable is None or not executable.is_file():
                asr["error_code"] = "ASR_EXECUTABLE_MISSING"
            elif not _local_model(local):
                # Unlike the legacy CLI preflight, a known downloadable name is
                # insufficient. The isolated GUI worker never requests a download.
                asr["error_code"] = "ASR_LOCAL_MODEL_MISSING"
        elif backend == "kotoba":
            local = Path(model)
            if not local.is_absolute() or not _local_model(local):
                asr["error_code"] = "ASR_LOCAL_MODEL_REQUIRED"
            elif any(importlib.util.find_spec(name) is None for name in ("faster_whisper", "ctranslate2")):
                asr["error_code"] = "ASR_DEPENDENCY_MISSING"
        else:
            asr["error_code"] = "ASR_BACKEND_UNSUPPORTED"
        device = asr["device"]
        if not asr["error_code"]:
            if device not in {"cpu", "cuda"}:
                asr["error_code"] = "ASR_DEVICE_UNSUPPORTED"
            elif device == "cuda" and not (runtime_probe or cuda_available)():
                asr["error_code"] = "ASR_CUDA_UNAVAILABLE"
            elif settings["transcribe_compute_type"] not in {"default", "auto", "int8", "int8_float16", "int8_float32", "int8_bfloat16", "int16", "float16", "float32", "bfloat16"}:
                asr["error_code"] = "ASR_COMPUTE_TYPE_UNSUPPORTED"
            elif device == "cpu" and settings["transcribe_compute_type"] in {"float16", "int8_float16", "bfloat16", "int8_bfloat16"}:
                asr["error_code"] = "ASR_DEVICE_COMPUTE_MISMATCH"
        asr["available"] = asr["error_code"] is None
    if translation_needed:
        endpoint = urlsplit(settings["translate_api"])
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            translation["error_code"] = "TRANSLATION_ENDPOINT_INVALID"
        else:
            try:
                preflight_model(settings["translate_api"], settings["translate_model"], client=client, timeout=5)
                translation["available"] = True
            except Exception:
                # Never export endpoint response text, private paths or tokens.
                translation["error_code"] = "TRANSLATION_MODEL_UNAVAILABLE"
    if conversion_needed:
        conversion["available"] = importlib.util.find_spec("opencc") is not None
        if not conversion["available"]:
            conversion["error_code"] = "OPENCC_DEPENDENCY_MISSING"
    return {"schema_version": "1.0", "plan_hash": plan.plan_hash,
            "available": all(stage["available"] for stage in (asr, translation, conversion, runtime)),
            "asr": asr, "translation": translation, "conversion": conversion, "runtime": runtime,
            "inference_proven": False}
