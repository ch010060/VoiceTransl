"""Transcribe one audio file with WhisperX (large-v3, batched, + forced alignment) and write an SRT.

Runs in its own `asmr-whisperx` conda env, so it imports nothing from the asmr pipeline.
One-time setup (PowerShell):
    conda create -n asmr-whisperx python=3.11 -y
    conda run -n asmr-whisperx python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    conda run -n asmr-whisperx python -m pip install whisperx
Usage:
    python scripts/whisperx_transcribe.py AUDIO OUT.srt [--batch_size 16]
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Same vocabulary prompt as the production pipeline (cli.DEFAULT_ASMR_INITIAL_PROMPT);
# copied because this env cannot import the pipeline.
ASMR_PROMPT = (
    "日本語の成人向けASMR音声作品です。"
    "サキュバスや魔法少女が耳舐め、耳かき、手コキ、フェラチオ、クンニで奉仕します。"
    "勃起したおちんちん、おちんぽを舌で舐め、搾精し射精させます。"
    "精液、愛液でシーツが汚れ、精気を吸い取ります。"
    "乳首、おっぱい、膣、亀頭、絶頂、中出し、潮吹き。"
    "気持ちいい、イく、出して。"
)


def _ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    # Alignment can leave a segment without start/end when no char could be aligned; drop those.
    kept = [s for s in segments if s.get("text", "").strip() and "start" in s and "end" in s]
    return "\n".join(
        f"{i}\n{_ts(s['start'])} --> {_ts(s['end'])}\n{s['text'].strip()}\n" for i, s in enumerate(kept, start=1)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("out_srt")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    import whisperx

    model = whisperx.load_model(
        "large-v3", "cuda", compute_type="float16", language="ja",
        asr_options={"initial_prompt": ASMR_PROMPT},
    )
    audio = whisperx.load_audio(args.audio)
    result = model.transcribe(audio, batch_size=args.batch_size, language="ja")
    align_model, align_meta = whisperx.load_align_model(language_code="ja", device="cuda")
    aligned = whisperx.align(result["segments"], align_model, align_meta, audio, "cuda", return_char_alignments=False)

    out = Path(args.out_srt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(segments_to_srt(aligned["segments"]), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
