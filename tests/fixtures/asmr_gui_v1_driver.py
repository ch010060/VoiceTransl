"""Synthetic GUI integration adapter; no models or external service are invoked.

This test-only program runs the real contained worker and durable protocol with
injected short SRT-producing adapters. It deliberately emits native-shaped ASR
progress and spends time in translation so GUI states can be inspected.
"""
from pathlib import Path
import json
import os
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import asmr_gui_worker


def synthetic_preflight(plan, project_root):
    stage = {"available": True, "required": True, "inference_proven": False}
    return {"schema_version": "1.0", "plan_hash": plan.plan_hash, "available": True, "inference_proven": False,
        "asr": dict(stage, backend=plan.settings["transcribe_backend"], device=plan.settings.get("device", "cuda")),
        "translation": dict(stage, required=plan.mode == "full", model_id="SYNTHETIC_NO_INFERENCE"),
        "conversion": dict(stage), "runtime": dict(stage)}


def synthetic_asr(*, audio_path, transcribe_srt_path, **kwargs):
    callback = kwargs.get("transcription_progress_callback")
    delay = 1.5 if "slow" in audio_path.stem else .03
    for ratio in (.15, .45, .8):
        if callback:
            callback(ratio, ratio * 10, 10.)
        time.sleep(delay)
    if "fail" in audio_path.stem:
        raise RuntimeError("SYNTHETIC_GUI_TRACK_FAILURE")
    transcribe_srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nSynthetic neutral audio fixture.\n",
        encoding="utf-8")
    return transcribe_srt_path


def synthetic_translation(source, target):
    time.sleep(5 if "slow" in Path(source).stem else .08)
    shutil.copyfile(source, target)
    return target


def synthetic_copy(source, target):
    shutil.copyfile(source, target)
    return target


def run(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    transcribe_only = False
    if arguments and arguments[0] == "run":
        document = json.loads(Path(arguments[arguments.index("--plan") + 1]).read_text(encoding="utf-8"))
        transcribe_only = document["plan"]["mode"] == "transcribe_only"
    if arguments and arguments[0] == "preflight":
        time.sleep(float(os.environ.get("ASMR_GUI_TEST_PREFLIGHT_DELAY", "0")))
    asmr_gui_worker.preflight = synthetic_preflight
    return asmr_gui_worker.main(arguments, adapters={
        "preflight": synthetic_preflight,
        "transcriber": synthetic_asr,
        "translator": None if transcribe_only else synthetic_translation,
        "opencc_converter": synthetic_copy,
    })


if __name__ == "__main__":
    raise SystemExit(run())
