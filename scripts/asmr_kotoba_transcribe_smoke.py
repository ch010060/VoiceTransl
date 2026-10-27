from __future__ import annotations

import argparse
from pathlib import Path
from time import monotonic

from asmr_transcribe_lib.kotoba_faster_whisper import (
    DEFAULT_KOTOBA_MODEL,
    KotobaFasterWhisperConfig,
    run_kotoba_faster_whisper_transcription,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asmr_kotoba_transcribe_smoke",
        description="Run one isolated Kotoba Python faster-whisper transcription smoke test.",
    )
    parser.add_argument("--audio", required=True, help="Audio file to transcribe.")
    parser.add_argument("--output", default=None, help="Target SRT path. Defaults to <audio stem>-kotoba-transcribe.srt.")
    parser.add_argument("--model", default=DEFAULT_KOTOBA_MODEL)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute_type", default="float16")
    parser.add_argument("--chunk_length", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audio = Path(args.audio)
    output = Path(args.output) if args.output else audio.with_name(f"{audio.stem}-kotoba-transcribe.srt")
    started = monotonic()
    print("Kotoba Python faster-whisper smoke")
    print(f"Audio: {audio}")
    print(f"Output: {output}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Compute type: {args.compute_type}")
    print(f"Chunk length: {args.chunk_length}")
    run_kotoba_faster_whisper_transcription(
        audio_path=audio,
        transcribe_srt_path=output,
        project_root=Path(__file__).resolve().parents[1],
        config=KotobaFasterWhisperConfig(
            model=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            chunk_length=args.chunk_length,
            condition_on_previous_text=False,
        ),
    )
    print(f"Done in {monotonic() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
