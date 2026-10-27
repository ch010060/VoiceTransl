from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable


DEFAULT_KOTOBA_TRANSFORMERS_MODEL = "kotoba-tech/kotoba-whisper-v2.1"


@dataclass(frozen=True)
class KotobaTransformersConfig:
    pipeline_task: str | None = None
    model: str = DEFAULT_KOTOBA_TRANSFORMERS_MODEL
    language: str = "ja"
    task: str = "transcribe"
    device: str = "cuda:0"
    torch_dtype: str = "float16"
    attn_implementation: str | None = "sdpa"
    batch_size: int = 16
    chunk_length_s: int = 15
    trust_remote_code: bool = True
    punctuator: bool | None = True
    add_punctuation: bool | None = None
    add_silence_start: float | None = None
    add_silence_end: float | None = None


PipelineFactory = Callable[..., Any]


def run_kotoba_transformers_transcription(
    *,
    audio_path: Path | str,
    transcribe_srt_path: Path | str,
    config: KotobaTransformersConfig,
    pipeline_factory: PipelineFactory | None = None,
) -> Path:
    audio = Path(audio_path).resolve()
    target_srt = Path(transcribe_srt_path).resolve()
    if target_srt.exists():
        target_srt.unlink()
    target_srt.parent.mkdir(parents=True, exist_ok=True)

    factory = pipeline_factory
    torch_module: Any | None = None
    if factory is None:
        try:
            torch_module = import_module("torch")
            transformers = import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "Kotoba Transformers adapter requires optional dependencies transformers and torch. "
                "Install them in the asmr conda environment before running this backend."
            ) from exc
        factory = transformers.pipeline

    pipeline_kwargs = _build_pipeline_kwargs(config, torch_module=torch_module)
    pipe = factory(**pipeline_kwargs)

    call_kwargs = _build_call_kwargs(config)
    result = pipe(str(audio), **call_kwargs)
    _write_chunks_as_srt(flatten_pipeline_chunks(result), target_srt)
    return target_srt


def flatten_pipeline_chunks(result: Any) -> list[tuple[float, float, str]]:
    if not isinstance(result, dict):
        return []
    chunks = result.get("chunks") or []
    flattened: list[tuple[float, float, str]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        timestamp = chunk.get("timestamp")
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
            continue
        start, end = timestamp
        text = str(chunk.get("text", "")).strip()
        if start is None or end is None or not text:
            continue
        flattened.append((float(start), float(end), text))
    return flattened


def format_srt_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _build_pipeline_kwargs(config: KotobaTransformersConfig, *, torch_module: Any | None) -> dict[str, Any]:
    torch_dtype: Any = config.torch_dtype
    if torch_module is not None and isinstance(config.torch_dtype, str):
        torch_dtype = getattr(torch_module, config.torch_dtype)

    kwargs: dict[str, Any] = {
        "model": config.model,
        "torch_dtype": torch_dtype,
        "device": config.device,
        "model_kwargs": {},
        "batch_size": config.batch_size,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.pipeline_task:
        kwargs = {"task": config.pipeline_task, **kwargs}
    if config.attn_implementation:
        kwargs["model_kwargs"] = {"attn_implementation": config.attn_implementation}
    if config.punctuator is not None:
        kwargs["punctuator"] = config.punctuator
    return kwargs


def _build_call_kwargs(config: KotobaTransformersConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "return_timestamps": True,
        "generate_kwargs": {"language": config.language, "task": config.task},
    }
    if config.chunk_length_s > 0:
        kwargs["chunk_length_s"] = config.chunk_length_s
    if config.add_punctuation is not None:
        kwargs["add_punctuation"] = config.add_punctuation
    if config.add_silence_start is not None:
        kwargs["add_silence_start"] = config.add_silence_start
    if config.add_silence_end is not None:
        kwargs["add_silence_end"] = config.add_silence_end
    return kwargs


def _write_chunks_as_srt(chunks: list[tuple[float, float, str]], target_srt: Path) -> None:
    lines: list[str] = []
    for index, (start, end, text) in enumerate(chunks, start=1):
        if lines:
            lines.append("")
        lines.extend(
            [
                str(index),
                f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}",
                text,
            ]
        )
    target_srt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
