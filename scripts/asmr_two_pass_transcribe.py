"""Experimental confidence-gated batched transcription; never used by the CLI.

Usage (asmr environment): python scripts/asmr_two_pass_transcribe.py AUDIO OUT.srt
Use --mode batched for the Python API control. Scores and retry timings are saved
beside OUT.srt. API segmentation/VAD may differ from the production XXL binary;
compare both controls before attributing a change to the retry strategy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time


def select_windows(segments: list[dict], duration: float, *, logprob: float = -0.8,
                   no_speech: float = 0.6, compression: float = 2.4,
                   padding: float = 1.0) -> list[tuple[float, float]]:
    """Select suspect speech and include whole captions touched by the padding.

    Gaps with no emitted segment have no confidence score and cannot be detected
    by this selector. These thresholds are experimental, not calibrated labels.
    """
    windows = []
    for segment in segments:
        if (segment["avg_logprob"] < logprob or segment["no_speech_prob"] > no_speech
                or segment["compression_ratio"] > compression):
            windows.append((max(0, segment["start"] - padding),
                            min(duration, segment["end"] + padding)))
    expanded = []
    for start, end in windows:
        while True:
            touched = [s for s in segments if s["start"] < end and start < s["end"]]
            bounds = (max(0, min([start] + [s["start"] for s in touched])),
                      min(duration, max([end] + [s["end"] for s in touched])))
            if bounds == (start, end):
                break
            start, end = bounds
        if end > start:
            expanded.append((start, end))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(expanded):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def splice_segments(original: list[dict], retries: list[tuple[float, float, list[dict]]]) -> list[dict]:
    """Retry timestamps are relative to the crop; preserve originals on empty retry."""
    result = list(original)
    for start, end, candidates in retries:
        replacement = []
        for item in candidates:
            segment = dict(item, start=max(start, start + item["start"]),
                           end=min(end, start + item["end"]))
            if not segment["text"].strip() or segment["end"] <= segment["start"]:
                continue
            if segment.get("words"):
                segment["words"] = [dict(w, start=max(start, start + w["start"]),
                    end=min(end, start + w["end"])) for w in segment["words"]
                    if min(end, start + w["end"]) > max(start, start + w["start"])]
            replacement.append(segment)
        if replacement:
            result = [s for s in result if not (s["start"] < end and start < s["end"])]
            result.extend(replacement)
    return sorted(result, key=lambda s: (s["start"], s["end"]))


def _timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    seconds, ms = divmod(milliseconds, 1000)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{sec:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    captions = []
    for segment in segments:
        if segment.get("words"):
            pending = []
            for word in segment["words"]:
                pending.append(word)
                if word["word"].rstrip().endswith(tuple("。！？!?")):
                    captions.append((pending[0]["start"], pending[-1]["end"],
                                     "".join(w["word"] for w in pending)))
                    pending = []
            if pending:
                captions.append((pending[0]["start"], pending[-1]["end"],
                                 "".join(w["word"] for w in pending)))
        else:
            captions.append((segment["start"], segment["end"], segment["text"]))
    blocks = []
    for start, end, text in captions:
        if text.strip() and round(end * 1000) > round(start * 1000):
            blocks.append(f"{len(blocks) + 1}\n{_timestamp(start)} --> {_timestamp(end)}\n{text.strip()}\n")
    return "\n".join(blocks)


def main(default_mode: str = "two-pass") -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["batched", "two-pass"], default=default_mode)
    parser.add_argument("--logprob", type=float, default=-0.8)
    parser.add_argument("--no-speech", type=float, default=0.6)
    parser.add_argument("--compression", type=float, default=2.4)
    args = parser.parse_args()

    # Import GPU dependencies only when executing, so unit tests stay CPU-only.
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from faster_whisper.audio import decode_audio

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from asmr_transcribe_lib.cli import DEFAULT_ASMR_INITIAL_PROMPT

    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    audio = decode_audio(str(args.audio), sampling_rate=16000)
    duration = len(audio) / 16000
    model = WhisperModel(str(root / "whisper-faster/faster-whisper-large-v3"),
                         device="cuda", compute_type="float16")
    generated, _ = BatchedInferencePipeline(model).transcribe(
        audio, language="ja", initial_prompt=DEFAULT_ASMR_INITIAL_PROMPT,
        batch_size=8, chunk_length=20, word_timestamps=True, vad_filter=True)
    batched = [asdict(s) for s in generated]
    batch_seconds = time.monotonic() - started
    windows = select_windows(batched, duration, logprob=args.logprob,
                             no_speech=args.no_speech, compression=args.compression)
    retries = []
    retry_started = time.monotonic()
    if args.mode == "two-pass":
        for start, end in windows:
            generated, _ = model.transcribe(
                audio[round(start * 16000):round(end * 16000)], language="ja",
                initial_prompt=DEFAULT_ASMR_INITIAL_PROMPT, word_timestamps=True,
                vad_filter=True, condition_on_previous_text=True)
            retries.append((start, end, [asdict(s) for s in generated]))
    result = splice_segments(batched, retries)
    retry_seconds = time.monotonic() - retry_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".batched.srt").write_text(segments_to_srt(batched), encoding="utf-8")
    args.output.write_text(segments_to_srt(result), encoding="utf-8")
    report = dict(audio=str(args.audio.resolve()), mode=args.mode, duration=duration,
        thresholds=dict(logprob=args.logprob, no_speech=args.no_speech, compression=args.compression),
        batch_seconds=batch_seconds, retry_seconds=retry_seconds,
        retry_audio_seconds=sum(end - start for start, end, _ in retries),
        selected_windows=windows, retried_windows=len(retries), batched=batched,
        retries=[dict(start=s, end=e, segments=items) for s, e, items in retries])
    args.output.with_suffix(".scores.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.mode}: {len(batched)} segments, {len(retries)} retries, "
          f"{batch_seconds + retry_seconds:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
