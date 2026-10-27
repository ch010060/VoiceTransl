from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

_RE_DURATION = re.compile(r"Processing audio with duration\s+([\d:]+\.[\d]+)")
_RE_SEGMENT = re.compile(r"Processing segment at\s+([\d:]+\.[\d]+)")


def _hmss_to_seconds(ts: str) -> float | None:
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except (ValueError, TypeError):
        pass
    return None

GUI_MODEL_PREFIX = "faster-whisper-"
KNOWN_FASTER_WHISPER_XXL_MODELS = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "large-v3-turbo",
    "turbo",
    "distil-large-v2",
    "distil-medium.en",
    "distil-small.en",
    "distil-large-v3",
    "distil-large-v3.5",
}


@dataclass(frozen=True)
class FasterWhisperConfig:
    model: str
    language: str
    compute_type: str = "float16"
    model_dir: Path | None = None
    vad_filter: bool = True
    word_timestamps: bool = True
    batched: bool = False
    batch_size: int = 16
    unmerged: bool = False
    initial_prompt: str | None = None
    reprompt: bool = False


def find_faster_whisper_exe(project_root: Path | str) -> Path | None:
    root = Path(project_root)
    candidates = [
        root / "whisper-faster" / "whisper-faster.exe",
        root / "Whisper-Faster" / "whisper-faster.exe",
        root / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def preflight_faster_whisper(project_root: Path | str) -> Path:
    exe = find_faster_whisper_exe(project_root)
    if exe is None:
        raise FileNotFoundError(
            "faster-whisper executable not found. Expected one of: "
            "whisper-faster/whisper-faster.exe, Whisper-Faster/whisper-faster.exe, "
            "or Faster-Whisper-XXL/faster-whisper-xxl.exe"
        )
    return exe


def find_faster_whisper_model_dir(project_root: Path | str) -> Path | None:
    root = Path(project_root)
    candidates = [
        root / "whisper-faster",
        root / "Whisper-Faster",
        root / "Faster-Whisper-XXL" / "_models",
        root / "Faster-Whisper-XXL",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def resolve_faster_whisper_model_arg(model: str) -> str:
    if model.startswith(GUI_MODEL_PREFIX):
        return model[len(GUI_MODEL_PREFIX) :]
    return model


def preflight_faster_whisper_model(project_root: Path | str, model: str) -> Path | None:
    preflight_faster_whisper(project_root)
    model_arg = resolve_faster_whisper_model_arg(model)

    model_dir = find_faster_whisper_model_dir(project_root)
    expected_folder_name = model if model.startswith(GUI_MODEL_PREFIX) else f"{GUI_MODEL_PREFIX}{model_arg}"
    expected_folder = model_dir / expected_folder_name if model_dir is not None else Path(project_root) / "whisper-faster" / expected_folder_name
    if expected_folder.exists() and expected_folder.is_dir():
        return expected_folder

    if model_arg in KNOWN_FASTER_WHISPER_XXL_MODELS:
        return None

    raise FileNotFoundError(
        "faster-whisper model folder not found. "
        f"Expected local CT2 model folder: {expected_folder}. "
        f"For GUI-style model '{model}', the XXL binary receives --model '{model_arg}'. "
        "Either place the model folder under whisper-faster or pass a built-in model such as large-v3-turbo."
    )


def build_faster_whisper_argv(
    exe_path: Path,
    audio_path: Path,
    output_dir: Path,
    config: FasterWhisperConfig,
) -> list[str]:
    argv = [
        str(exe_path),
        str(audio_path),
        "--beep_off",
        "--verbose",
        "True",
        "--model",
        resolve_faster_whisper_model_arg(config.model),
        *(_model_dir_args(config.model_dir)),
        "--task",
        "transcribe",
        "--language",
        config.language,
        "--output_format",
        "srt",
        "--output_dir",
        str(output_dir),
        "--compute_type",
        config.compute_type,
        "--vad_filter",
        "True" if config.vad_filter else "False",
        "--word_timestamps",
        "True" if config.word_timestamps else "False",
    ]
    if config.batched:
        argv += ["--batched", "--batch_size", str(config.batch_size)]
        if config.unmerged:
            argv += ["--unmerged"]
    if config.initial_prompt is not None:
        argv += ["--initial_prompt", config.initial_prompt]
        if config.reprompt:
            argv += ["--reprompt", "True"]
    return argv


def _model_dir_args(model_dir: Path | None) -> list[str]:
    if model_dir is None:
        return []
    return ["--model_dir", str(model_dir)]


def run_faster_whisper_transcription(
    *,
    audio_path: Path | str,
    transcribe_srt_path: Path | str,
    project_root: Path | str,
    config: FasterWhisperConfig,
    transcription_progress_callback: Callable[[float, float, float], None] | None = None,
    subprocess_run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path:
    project_root_path = Path(project_root).resolve()
    audio = Path(audio_path).resolve()
    target_srt = Path(transcribe_srt_path).resolve()
    exe = preflight_faster_whisper(project_root_path)

    if target_srt.exists():
        target_srt.unlink()

    target_srt.parent.mkdir(parents=True, exist_ok=True)
    temp_root = project_root_path / ".asmr_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="asmr-fw-", dir=str(temp_root)) as tmp_dir_name:
            output_dir = Path(tmp_dir_name)
            argv = build_faster_whisper_argv(exe, audio, output_dir, config)
            if transcription_progress_callback is not None:
                returncode, combined_stdout = _run_with_progress(argv, project_root_path, transcription_progress_callback)
                stdout, stderr = combined_stdout, ""
            else:
                completed = subprocess_run(
                    argv,
                    cwd=project_root_path,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                returncode = getattr(completed, "returncode", None)
                stdout = getattr(completed, "stdout", "") or ""
                stderr = getattr(completed, "stderr", "") or ""
            produced_srt = _find_transcription_output(output_dir, audio)
            if produced_srt is None:
                if returncode not in (None, 0):
                    raise RuntimeError(
                        "faster-whisper exited without SRT output "
                        f"(return code {returncode}). stdout tail: {_tail(stdout)} stderr tail: {_tail(stderr)}"
                    )
                raise FileNotFoundError(
                    f"faster-whisper transcription SRT output not found in {output_dir} for {audio.name}"
                )
            shutil.move(str(produced_srt), str(target_srt))
    finally:
        _remove_empty_dir(temp_root)
    return target_srt


def _find_transcription_output(output_dir: Path, audio_path: Path) -> Path | None:
    candidates = [
        output_dir / f"{audio_path.stem}.srt",
        output_dir / f"{audio_path.name}.srt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    srt_files = sorted(output_dir.rglob("*.srt"), key=lambda path: path.as_posix().casefold())
    if len(srt_files) == 1:
        return srt_files[0]
    return None


def _remove_empty_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def _run_with_progress(
    argv: list[str],
    cwd: Path,
    callback: Callable[[float, float, float], None],
) -> tuple[int | None, str]:
    lines: list[str] = []
    total_s: float | None = None
    with subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if total_s is None:
                m = _RE_DURATION.search(line)
                if m:
                    total_s = _hmss_to_seconds(m.group(1))
            elif total_s and total_s > 0:
                m = _RE_SEGMENT.search(line)
                if m:
                    seg = _hmss_to_seconds(m.group(1))
                    if seg is not None:
                        callback(min(seg / total_s, 1.0), seg, total_s)
        returncode = proc.wait()
    return returncode, "".join(lines)


def _tail(text: str, max_chars: int = 1000) -> str:
    return text[-max_chars:].replace("\r", "\\r").replace("\n", "\\n")
