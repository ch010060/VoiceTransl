"""Transcribe one audio file with NVIDIA Parakeet TDT-CTC 0.6B ja and write an SRT.

Runs in the separate `asmr-parakeet` conda env (NeMo), so it imports nothing from the
asmr pipeline. Usage:
    python scripts/parakeet_transcribe.py AUDIO OUT.srt [--model nvidia/parakeet-tdt_ctc-0.6b-ja]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def _ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


SENTENCE_END = set("。？！?!")


def split_chars(chars: list[dict], *, max_gap: float = 0.6, max_chars: int = 40, max_seconds: float = 8.0) -> list[tuple[float, float, str]]:
    # NeMo's own "segment" only splits on ASCII . ? ! so Japanese comes back as one block.
    # Build subtitle blocks from char timestamps: break at sentence end, a pause, or a length cap.
    segments: list[tuple[float, float, str]] = []
    cur: list[dict] = []

    def flush() -> None:
        text = "".join(c["char"] for c in cur).strip()
        if text:
            segments.append((cur[0]["start"], cur[-1]["end"], text))
        cur.clear()

    # NeMo 3.0 gives each char as a list of strings, e.g. ['心'] or [''].
    chars = [{**c, "char": c["char"] if isinstance(c["char"], str) else "".join(c["char"])} for c in chars]
    for c in chars:
        if cur and (
            c["start"] - cur[-1]["end"] > max_gap
            or len(cur) >= max_chars
            or c["end"] - cur[0]["start"] > max_seconds
        ):
            flush()
        cur.append(c)
        if c["char"] in SENTENCE_END:
            flush()
    if cur:
        flush()
    return segments


def to_mono_16k(audio: Path, tmp: Path) -> Path:
    # NeMo expects 16 kHz mono; ffmpeg is already a pipeline dependency.
    wav = tmp / "audio16k.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(audio), "-ac", "1", "-ar", "16000", str(wav)], check=True)
    return wav


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("out_srt")
    parser.add_argument("--model", default="nvidia/parakeet-tdt_ctc-0.6b-ja")
    args = parser.parse_args()

    import torch
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(args.model).to("cuda").eval()
    with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
        wav = to_mono_16k(Path(args.audio), Path(tmp))
        # ponytail: whole file in one pass; add chunking if long tracks run out of VRAM
        hyp = model.transcribe([str(wav)], timestamps=True, batch_size=1)[0]

    segments = split_chars(hyp.timestamp.get("char") or [])
    out = Path(args.out_srt)
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks = [
        f"{i}\n{_ts(start)} --> {_ts(end)}\n{text}\n"
        for i, (start, end, text) in enumerate(segments, start=1)
    ]
    out.write_text("\n".join(blocks), encoding="utf-8")
    print(f"wrote {len(blocks)} blocks", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
