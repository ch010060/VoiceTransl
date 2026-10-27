from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


DEFAULT_KOTOBA_MODEL = "kotoba-tech/kotoba-whisper-v2.0-faster"


@dataclass(frozen=True)
class KotobaFasterWhisperConfig:
    model: str = DEFAULT_KOTOBA_MODEL
    language: str = "ja"
    device: str = "cuda"
    compute_type: str = "float16"
    chunk_length: int = 15
    condition_on_previous_text: bool = False


def preflight_kotoba_faster_whisper() -> None:
    _load_whisper_model_cls()


def run_kotoba_faster_whisper_transcription(
    *,
    audio_path: Path | str,
    transcribe_srt_path: Path | str,
    project_root: Path | str,
    config: KotobaFasterWhisperConfig,
    whisper_model_cls: type[Any] | None = None,
) -> Path:
    del project_root
    audio = Path(audio_path).resolve()
    target_srt = Path(transcribe_srt_path).resolve()
    if target_srt.exists():
        target_srt.unlink()
    target_srt.parent.mkdir(parents=True, exist_ok=True)

    model_cls = whisper_model_cls if whisper_model_cls is not None else _load_whisper_model_cls()
    model = model_cls(
        config.model,
        device=config.device,
        compute_type=config.compute_type,
    )
    segments, _info = model.transcribe(
        str(audio),
        language=config.language,
        chunk_length=config.chunk_length,
        condition_on_previous_text=config.condition_on_previous_text,
    )
    _write_segments_as_srt(segments, target_srt)
    return target_srt


def format_srt_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_segments_as_srt(segments: Any, target_srt: Path) -> None:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = str(getattr(segment, "text", "")).strip()
        if not text:
            continue
        if lines:
            lines.append("")
        lines.extend(
            [
                str(index),
                f"{format_srt_timestamp(float(segment.start))} --> {format_srt_timestamp(float(segment.end))}",
                text,
            ]
        )
    target_srt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _load_whisper_model_cls() -> type[Any]:
    try:
        module = import_module("faster_whisper")
    except ImportError as exc:
        raise RuntimeError(
            "Kotoba Python adapter requires the optional dependency faster-whisper. "
            "Install it in the asmr conda environment with: pip install faster-whisper"
        ) from exc
    return module.WhisperModel
