from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

from asmr_transcribe_lib.kotoba_transformers import (
    DEFAULT_KOTOBA_TRANSFORMERS_MODEL,
    KotobaTransformersConfig,
    run_kotoba_transformers_transcription,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asmr_kotoba_transformers_smoke",
        description="Run one isolated Kotoba Transformers transcription smoke test.",
    )
    parser.add_argument("--audio", required=True, help="Audio file to transcribe.")
    parser.add_argument("--output", default=None, help="Target SRT path. Defaults to <audio stem>-kotoba-transformers.srt.")
    parser.add_argument("--model", default=DEFAULT_KOTOBA_TRANSFORMERS_MODEL)
    parser.add_argument("--pipeline_task", default=None, help="Optional explicit Transformers pipeline task.")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_length_s", type=int, default=15)
    parser.add_argument("--punctuator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_punctuation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--add_silence_start", type=float, default=None)
    parser.add_argument("--add_silence_end", type=float, default=None)
    parser.add_argument("--metrics", default=None, help="Optional JSON metrics output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audio = Path(args.audio)
    output = Path(args.output) if args.output else audio.with_name(f"{audio.stem}-kotoba-transformers.srt")
    config = KotobaTransformersConfig(
        model=args.model,
        pipeline_task=args.pipeline_task,
        language=args.language,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation or None,
        batch_size=args.batch_size,
        chunk_length_s=args.chunk_length_s,
        punctuator=args.punctuator,
        add_punctuation=args.add_punctuation,
        add_silence_start=args.add_silence_start,
        add_silence_end=args.add_silence_end,
    )

    started = monotonic()
    print("Kotoba Transformers smoke")
    print(f"Audio: {audio}")
    print(f"Output: {output}")
    print(f"Model: {config.model}")
    print(f"Device: {config.device}")
    print(f"Torch dtype: {config.torch_dtype}")
    print(f"Attention: {config.attn_implementation}")
    print(f"Batch size: {config.batch_size}")
    print(f"Chunk length: {config.chunk_length_s}")
    print(f"Punctuator: {config.punctuator}")

    run_kotoba_transformers_transcription(
        audio_path=audio,
        transcribe_srt_path=output,
        config=config,
    )
    elapsed = monotonic() - started
    text = output.read_text(encoding="utf-8") if output.exists() else ""
    block_count = text.count("\n\n") + (1 if text.strip() else 0)
    metrics = {
        "audio": str(audio),
        "output": str(output),
        "model": config.model,
        "elapsed_seconds": elapsed,
        "srt_blocks": block_count,
        "srt_chars": len(text),
    }
    print(f"Done in {elapsed:.2f}s")
    print(f"SRT blocks: {block_count}")
    print(f"SRT chars: {len(text)}")
    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
