"""Opt-in local Phase 1C smoke; never collected by pytest or run implicitly.

Prepare creates offline SAPI audio only. Execution requires the operator's explicit
baseline acknowledgement and uses the real Controller and unmodified worker.
All artifacts are retained. No model download or external service lifecycle call.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
import wave

PROJECT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (Path(tempfile.gettempdir()) / "VoiceTransl" / "Tests").resolve()
OWNER_FILE = "real-smoke-owner.json"
sys.path.insert(0, str(PROJECT))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def endpoint_status():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=3) as response:
            return {"reachable": True, "status": response.status, "models": json.load(response)}
    except Exception as exc:
        return {"reachable": False, "error": str(exc), "type": type(exc).__name__}


def service_processes():
    # Read-only inventory; do not acquire, stop or add external PIDs to our Job.
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'llama|sakura' } | Select-Object ProcessId,Name,CreationDate | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, check=False)
    return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def prepare(base):
    base.mkdir(parents=True, exist_ok=False)
    write_json(base / OWNER_FILE, {"kind": "VoiceTransl real smoke", "project": str(PROJECT),
        "base": str(base), "token": uuid.uuid4().hex})
    library = base / "library"
    work = library / "Neutral Weather Test"
    work.mkdir(parents=True)
    script = base / "generate_neutral_audio.ps1"
    script.write_text(r'''param([string]$Work)
Add-Type -AssemblyName System.Speech
$speech = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = @($speech.GetInstalledVoices() | ForEach-Object { [PSCustomObject]@{Name=$_.VoiceInfo.Name; Culture=$_.VoiceInfo.Culture.Name} })
$voice = $voices | Where-Object { $_.Culture -like 'ja-*' } | Select-Object -First 1
$language = 'ja'
if (-not $voice) { $voice = $voices | Where-Object { $_.Culture -like 'en-*' } | Select-Object -First 1; $language = 'en' }
if (-not $voice) { throw 'No installed Japanese or English offline voice' }
$speech.SelectVoice($voice.Name)
$speech.Rate = 0
if ($language -eq 'ja') {
    $texts = @('今日は晴れです。青い空に白い雲が見えます。机の上に本があります。', 'これは短い音声のテストです。窓の外には緑の木があります。午後に図書館へ行きます。')
} else {
    $texts = @('Today the weather is clear. White clouds move across the blue sky. A book is on the table.', 'This is a short audio test. A green tree stands outside the window. We will visit the library this afternoon.')
}
$names = @('01_weather.wav','02_library.wav')
for ($index = 0; $index -lt 2; $index++) {
    $speech.SetOutputToWaveFile((Join-Path $Work $names[$index]))
    $speech.Speak($texts[$index])
    $speech.SetOutputToNull()
}
$speech.Dispose()
[PSCustomObject]@{InstalledVoices=$voices; Voice=$voice; SourceLanguage=$language; Texts=$texts} | ConvertTo-Json -Depth 4
''', encoding="utf-8-sig")
    generated = subprocess.run(["powershell", "-NoProfile", "-File", str(script), "-Work", str(work)], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    (base / "sapi_stdout.txt").write_text(generated.stdout, encoding="utf-8")
    (base / "sapi_stderr.txt").write_text(generated.stderr, encoding="utf-8")
    if generated.returncode:
        raise RuntimeError("Offline SAPI audio generation failed; see retained logs")
    voice = json.loads(generated.stdout)
    duplicate = work / "SEなし" / "01_weather.wav"
    duplicate.parent.mkdir()
    shutil.copyfile(work / "01_weather.wav", duplicate)
    settings = {"src_lang": voice["SourceLanguage"], "no_asmr_prompt": True}
    from scripts.asmr_transcribe_lib.gui_preflight import cuda_available
    from scripts.asmr_transcribe_lib.local_faster_whisper import find_faster_whisper_exe, find_faster_whisper_model_dir
    model_dir = find_faster_whisper_model_dir(PROJECT)
    inventory = []
    for audio in sorted(library.rglob("*.wav")):
        with wave.open(str(audio)) as source:
            info = {"frames": source.getnframes(), "rate": source.getframerate(), "channels": source.getnchannels(), "seconds": source.getnframes() / source.getframerate()}
        inventory.append({"path": str(audio), "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(), **info})
    prerequisites = {"created_at": datetime.now(timezone.utc).isoformat(), "python": sys.executable, "controller_process_id": os.getpid(), "fixture": "self-generated neutral offline SAPI speech; no user library material", "voice": voice, "settings_override": settings, "cuda_available": cuda_available(), "xxl_executable": str(find_faster_whisper_exe(PROJECT)), "local_model_assets": {name: (model_dir / "faster-whisper-large-v3" / name).stat().st_size for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json")}, "audio_inventory": inventory, "sakura": endpoint_status(), "external_service_processes": service_processes()}
    write_json(base / "prerequisites.json", prerequisites)
    write_json(base / "settings.json", settings)
    return prerequisites


def clone_audio(base, name):
    library = base / name
    if library.exists():
        raise FileExistsError(library)
    for source in (base / "library").rglob("*.wav"):
        target = library / source.relative_to(base / "library")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return library


def process_images(pids):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    images = {}
    for pid in pids:
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            images[str(pid)] = None
            continue
        try:
            buffer, size = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
            images[str(pid)] = buffer.value if kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)) else None
        finally:
            kernel.CloseHandle(handle)
    return images


def japanese_translation_only(base):
    """Explicit authored subtitle test; this is never ASR or resume provenance."""
    from scripts.asmr_transcribe_lib.translation import translate_srt_to_simplified
    from scripts.asmr_transcribe_lib.opencc_utils import convert_srt_file_opencc
    from scripts.asmr_transcribe_lib.srt_utils import parse_srt_file, validate_same_timing
    from scripts.asmr_transcribe_lib.gui_plan import validate_srt

    directory = base / "japanese_translation_only"
    directory.mkdir(exist_ok=False)
    source = directory / "authored_test_ja.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,500\n今日は晴れです。\n\n2\n00:00:02,000 --> 00:00:04,000\n午後に図書館へ行きます。\n", encoding="utf-8")
    record = {"scenario": "japanese-translation-only", "fixture": "Explicit TEST-authored two-line Japanese SRT; not ASR, not provenance, not full Japanese end-to-end inference", "sakura_before": endpoint_status(), "external_before": service_processes(), "settings": {"model": "sakura-14b-qwen2.5-v1.0-q4km.gguf", "endpoint": "http://127.0.0.1:8080/v1", "max_blocks": 20}}
    started = time.monotonic()
    translated = translate_srt_to_simplified(source, directory / "actual_sakura_zh_cn.srt", base_url=record["settings"]["endpoint"], model=record["settings"]["model"], max_blocks=20)
    traditional = convert_srt_file_opencc(translated, directory / "actual_opencc_zh_tw.srt")
    blocks = parse_srt_file(source)
    record["outputs"] = []
    for path in (source, translated, traditional):
        validate_srt(path)
        candidate = parse_srt_file(path)
        validate_same_timing(blocks, candidate)
        record["outputs"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "texts": ["\n".join(block.text) for block in candidate], "timestamps": [block.timestamp for block in candidate], "valid_srt": True, "same_timing": True})
    record.update(sakura_after=endpoint_status(), external_after=service_processes(), elapsed_seconds=time.monotonic() - started)
    write_json(base / "japanese-translation-only.json", record)
    return record


def execute(base, scenario, timeout):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from scripts.asmr_gui.controller import Controller
    from scripts.asmr_transcribe_lib.gui_plan import validate_srt

    app = QApplication.instance() or QApplication([])
    settings = json.loads((base / "settings.json").read_text(encoding="utf-8"))
    state_root = base / "state"
    library = base / "library"
    if scenario == "stop":
        library = clone_audio(base, "stop_library")
    elif scenario == "partial-resume":
        library = clone_audio(base, "partial_resume_library")
    elif scenario == "full-stop":
        library = clone_audio(base, "full_stop_library")
    elif scenario == "full-partial-resume":
        library = clone_audio(base, "full_partial_resume_library")
    elif scenario == "full-partial-shortgrace":
        library = clone_audio(base, "full_partial_shortgrace_library")
    elif scenario == "full-common-skip":
        library = base / "full_common_skip_library"
        work = library / "Neutral Subtitle Presence Test"
        work.mkdir(parents=True, exist_ok=False)
        originals = base / "library" / "Neutral Weather Test"
        for name, source in (("01_srt.wav", "01_weather.wav"), ("02_ass.wav", "02_library.wav"), ("03_lrc.wav", "01_weather.wav")):
            shutil.copyfile(originals / source, work / name)
        shutil.copyfile(originals / "01_weather-transcribe.srt", work / "01_srt.srt")
        (work / "02_ass.ass").write_text("[Script Info]\nTitle: TEST-authored subtitle presence fixture, not model output\nScriptType: v4.00+\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,TEST authored neutral presence check\n", encoding="utf-8")
        (work / "03_lrc.lrc").write_text("[ti:TEST-authored subtitle presence fixture, not model output]\n[00:00.00]TEST authored neutral presence check\n", encoding="utf-8")
    elif scenario == "full-restart":
        prior = json.loads((base / "full-stop.json").read_text(encoding="utf-8"))
        if not prior.get("cleanup", {}).get("cleanup_verified"):
            raise RuntimeError("Full restart requires prior verified cleanup")
        library = clone_audio(base, "full_restart_library")
    elif scenario == "full-check":
        library = clone_audio(base, "full_check_library")
    elif scenario == "skip":
        # Explicit legacy-output fixture derived from this smoke's real ASR.
        # It is deliberately not certified as translation or given provenance.
        from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan
        original = build_execution_plan(library, mode="transcribe_only", settings=settings)
        library = clone_audio(base, "skip_library")
        target_plan = build_execution_plan(library, mode="transcribe_only", settings=settings)
        originals = {track.task.audio_path.relative_to(original.root): track for track in original.tracks}
        for track in target_plan.tracks:
            source = originals[track.task.audio_path.relative_to(library)].task.transcribe_srt_path
            validate_srt(source)
            target = track.task.final_srt_path
            if not target.exists():
                shutil.copyfile(source, target)
    elif scenario == "restart":
        prior = json.loads((base / "stop.json").read_text(encoding="utf-8"))
        if not prior.get("cleanup", {}).get("cleanup_verified"):
            raise RuntimeError("Restart requires prior verified cleanup")
        library = clone_audio(base, "restart_library")
    controller_options = {}
    if scenario == "full-partial-shortgrace":
        from scripts.asmr_gui_runtime import ManagedWorker
        controller_options["worker_factory"] = lambda *args, **kwargs: ManagedWorker(*args, cooperative_timeout=.1, cleanup_timeout=5, **kwargs)
    controller = Controller(state_root=state_root, project_root=PROJECT, **controller_options)
    evidence = {"scenario": scenario, "controller_process_id": os.getpid(), "library": str(library), "states": [], "logs": [], "completions": [], "job_samples": [], "sakura_before": endpoint_status(), "external_before": service_processes(), "started_at": datetime.now(timezone.utc).isoformat()}
    if scenario == "skip":
        evidence["skip_fixture"] = "New temporary library only: final-named existing SRT bytes copied from this smoke's actual transcribe-only outputs; unverified legacy, no translation/provenance claim"
    if scenario == "full-partial-shortgrace":
        evidence["test_only_lifecycle_override"] = {"cooperative_timeout": .1, "cleanup_timeout": 5, "purpose": "Controlled positive partial-resume evidence; production default 5-second grace tested separately and unchanged; no model/performance configuration override"}
    if scenario == "full-common-skip":
        evidence["common_subtitle_fixture"] = "SRT bytes copied from this smoke's real ASR output; ASS/LRC explicitly TEST-authored format-presence fixtures, not model output; no inference expected"
        evidence["fixtures_before"] = {str(path): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mtime_ns": path.stat().st_mtime_ns} for path in library.rglob("*") if path.is_file()}
    controller.state_changed.connect(lambda state: evidence["states"].append({"state": state, "seconds": time.monotonic() - started}))
    controller.log_message.connect(evidence["logs"].append)
    controller.completed.connect(evidence["completions"].append)
    started = time.monotonic()
    stop_sent = False
    last_job = None
    last_stages = None

    def wait_idle():
        nonlocal stop_sent, last_job, last_stages
        deadline = time.monotonic() + timeout
        while controller.busy:
            app.processEvents()
            stages = tuple(sorted((key, track.stage, track.status) for key, track in controller.run_model.tracks.items()))
            if stages != last_stages:
                evidence.setdefault("stage_samples", []).append({"seconds": time.monotonic() - started, "tracks": stages})
                last_stages = stages
            if scenario == "full-stop" and not stop_sent and any(track.stage == "translation" for track in controller.run_model.tracks.values()):
                evidence["stop_request"] = {"seconds": time.monotonic() - started, "state": controller.state, "stage": "translation"}
                stop_sent = controller.stop()
            with controller._owned_lock:
                managed = controller._owned
            if managed is not None and getattr(managed, "plan", None) is not None and managed.job is not None and managed.job.handle is not None:
                try:
                    pids = managed.job.process_ids()
                    sample = (managed.run_id, tuple(pids))
                    if sample != last_job:
                        evidence["job_samples"].append({"run_id": managed.run_id, "job_name": managed.job.name, "pids": pids, "images": process_images(pids), "active_count": managed.job.active_count(), "seconds": time.monotonic() - started})
                        last_job = sample
                    # Bootstrap + production worker + actual native ASR child.
                    wants_stop = (scenario == "stop" or (scenario == "partial-resume" and controller.run_model.summary().get("transcribed", 0) >= 1) or (scenario in {"full-partial-resume", "full-partial-shortgrace"} and controller.run_model.summary().get("success", 0) >= 1))
                    if wants_stop and not stop_sent and any(Path(image).name.lower() == "faster-whisper-xxl.exe" for image in process_images(pids).values() if image):
                        evidence["stop_request"] = {"seconds": time.monotonic() - started, "pids": pids, "state": controller.state}
                        stop_sent = controller.stop()
                except OSError:
                    pass  # Background lifecycle may have just closed this handle.
            if time.monotonic() > deadline:
                evidence["timeout"] = True
                controller.stop()
                deadline = time.monotonic() + 20
                while controller.busy and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(.02)
                break
            time.sleep(.02)
        app.processEvents()

    try:
        full_mode = scenario in {"full", "full-stop", "full-partial-resume", "full-partial-shortgrace", "full-restart", "full-skip", "full-common-skip", "full-check"}
        controller.apply_settings(settings, "full" if full_mode else "transcribe_only")
        controller.scan(library)
        wait_idle()
        evidence["scan_state"] = controller.state
        plan = controller.run_model.plan
        if plan:
            evidence["plan"] = {"plan_hash": plan.plan_hash, "settings": plan.settings, "issues": [issue.code for issue in plan.issues], "works": [{"work_id": work.work_id, "track_ids": work.track_ids} for work in plan.works], "tracks": [{"track_id": track.track_id, "path": str(track.task.audio_path), "action": track.task.action, "dependencies": track.dependencies, "representative": track.representative_track_id, "verification": track.verification} for track in plan.tracks]}
        if controller.state == "READY":
            if scenario in {"full-check", "resume-check"}:
                evidence["accepted"] = controller.check_backends()
            else:
                evidence["accepted"] = controller.start()
            wait_idle()
        evidence.update(state=controller.state, error=controller.last_error, backend=controller.last_backend, cleanup=controller.last_cleanup, run_directory=str(controller.last_run_dir), model_summary=controller.run_model.summary(), stop_sent=stop_sent)
        if scenario in {"transcribe-resume", "partial-resume", "full-partial-resume", "full-partial-shortgrace"} and controller.last_cleanup and controller.last_cleanup.get("cleanup_verified") and (evidence["model_summary"].get("transcribed", 0) or evidence["model_summary"].get("success", 0)):
            evidence["resume"] = {"first_run_directory": str(controller.last_run_dir), "first_cleanup": controller.last_cleanup}
            controller.scan(library)
            wait_idle()
            resumed_plan = controller.run_model.plan
            evidence["resume"].update(scan_state=controller.state, scan_error=controller.last_error, issue_codes=[issue.code for issue in resumed_plan.issues] if resumed_plan else None)
            if controller.state == "READY" and resumed_plan and resumed_plan.executable:
                evidence["resume"]["accepted"] = controller.start()
                wait_idle()
            evidence["resume"].update(state=controller.state, error=controller.last_error, cleanup=controller.last_cleanup, run_directory=str(controller.last_run_dir), model_summary=controller.run_model.summary())
        evidence["outputs"] = []
        for output in sorted(library.rglob("*.srt")):
            try:
                validate_srt(output)
                valid, error = True, None
            except Exception as exc:
                valid, error = False, type(exc).__name__
            evidence["outputs"].append({"path": str(output), "valid_srt": valid, "validation_error": error, "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        evidence["sakura_after"] = endpoint_status()
        evidence["external_after"] = service_processes()
        if scenario == "full-common-skip":
            evidence["fixtures_after"] = {str(path): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mtime_ns": path.stat().st_mtime_ns} for path in library.rglob("*") if path.is_file()}
            evidence["all_original_bytes_and_mtimes_unchanged"] = evidence["fixtures_before"] == evidence["fixtures_after"]
            evidence["ass_lrc_srt_not_created"] = not (library / "Neutral Subtitle Presence Test" / "02_ass.srt").exists() and not (library / "Neutral Subtitle Presence Test" / "03_lrc.srt").exists()
        evidence["elapsed_seconds"] = time.monotonic() - started
        write_json(base / (scenario + ".json"), evidence)
    finally:
        if controller.busy:
            controller.stop()
            wait_idle()
        controller.request_close()
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
        help="Retained owned real-smoke-* directory under %%TEMP%%/VoiceTransl/Tests; --prepare defaults to a new UUID")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--scenario", choices=["transcribe", "transcribe-resume", "partial-resume", "full", "full-stop", "full-partial-resume", "full-partial-shortgrace", "full-restart", "full-skip", "full-common-skip", "full-check", "japanese-translation-only", "stop", "restart", "resume-check", "skip"])
    parser.add_argument("--baseline-green", action="store_true", help="Operator confirms full synthetic/native baseline passed before real work")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.base is None and not args.prepare:
        parser.error("Execution requires --base from the previous --prepare result")
    base = (args.base or ARTIFACT_ROOT / ("real-smoke-" + uuid.uuid4().hex)).resolve()
    if base.parent != ARTIFACT_ROOT or not base.name.startswith("real-smoke-") or base.is_relative_to(PROJECT):
        parser.error("base must be a real-smoke-* directory directly under %TEMP%/VoiceTransl/Tests, outside this repository")
    if args.prepare:
        result = prepare(base)
        print(json.dumps({"prepared": str(base), "settings": result["settings_override"], "audio": result["audio_inventory"], "sakura": result["sakura"]}, indent=2))
        return
    if not args.baseline_green or not args.scenario:
        parser.error("Execution requires --scenario and explicit --baseline-green")
    try:
        owner = json.loads((base / OWNER_FILE).read_text(encoding="utf-8"))
        if owner["kind"] != "VoiceTransl real smoke" or owner["project"] != str(PROJECT) or owner["base"] != str(base):
            raise ValueError("fixture ownership does not match")
        uuid.UUID(hex=owner["token"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"base is not an owned prepared smoke fixture: {exc}")
    result = japanese_translation_only(base) if args.scenario == "japanese-translation-only" else execute(base, args.scenario, args.timeout)
    print(json.dumps({key: result.get(key) for key in ("scenario", "state", "error", "run_directory", "model_summary", "cleanup", "outputs", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
