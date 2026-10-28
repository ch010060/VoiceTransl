"""A/B benchmark for transcription settings. Prints counts and timings only.

Usage:
    python scripts/asmr_transcribe_bench.py --path W:\\bench_audio --variants base parallel2 batched4 batched8
    python scripts/asmr_transcribe_bench.py --path W:\\bench_audio --variants parakeet --reuse_base
    python scripts/asmr_transcribe_bench.py --path W:\\bench_audio --variants parakeet --rescore

"base" is the production setting (large-v3 + ASMR prompt) and the reference for every
metric. Variants:
  base_rerun        same as base; shows the run-to-run noise floor
  vad_mid/strict    stricter voice detection (tested 2026-10: no gain)
  parallel2         base settings, 2 files transcribed at once; compare total time
  batched4/8        Faster-Whisper-XXL batched mode with small batches, segments kept merged
  batched8_sent     batched8 + --sentence (new subtitle per sentence)
  batched8_split    batched8 + --sentence, 1 s pause = break, max 40 chars per subtitle
  batched8_c20/c15  batched8_sent with 20 s / 15 s windows instead of 30 s
  parakeet          NVIDIA Parakeet in the asmr-parakeet env
  whisperx          WhisperX large-v3 + forced alignment in the asmr-whisperx env
--reuse_base skips re-running base; --rescore skips all transcription and recomputes the
table from SRTs already in --out. SRTs are kept in .asmr_bench_out/<variant>/.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asmr_transcribe_lib import local_faster_whisper as lf  # noqa: E402
from asmr_transcribe_lib.cli import DEFAULT_ASMR_INITIAL_PROMPT, DEFAULT_TRANSCRIBE_MODEL  # noqa: E402
from asmr_transcribe_lib.srt_utils import parse_srt_file  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}

# extra: Faster-Whisper-XXL flags added to the production command
# workers: files transcribed at the same time (total wall time is what counts)
# runner: (conda env, adapter script) for a non-XXL engine
VARIANTS: dict[str, dict] = {
    "base": {},
    "base_rerun": {},
    "vad_mid": {"extra": ["--vad_threshold", "0.6", "--vad_min_silence_duration_ms", "700", "--vad_speech_pad_ms", "250"]},
    "vad_strict": {"extra": [
        "--vad_threshold", "0.75", "--vad_min_speech_duration_ms", "250",
        "--vad_min_silence_duration_ms", "700", "--vad_speech_pad_ms", "250",
        "--no_speech_threshold", "0.5", "--hallucination_silence_threshold", "1.5",
    ]},
    "parallel2": {"workers": 2},
    "batched4": {"extra": ["--batched", "--batch_size", "4"]},
    "batched8": {"extra": ["--batched", "--batch_size", "8"]},
    # batched output comes back as ~26 s blocks; re-split it into subtitles via word timestamps
    "batched8_sent": {"extra": ["--batched", "--batch_size", "8", "--sentence"]},
    # shorter batched windows: less audio decoded per independent chunk (default 30 s)
    "batched8_c20": {"extra": ["--batched", "--batch_size", "8", "--sentence", "--chunk_length", "20"]},
    "batched8_c15": {"extra": ["--batched", "--batch_size", "8", "--sentence", "--chunk_length", "15"]},
    "batched8_split": {"extra": [
        "--batched", "--batch_size", "8", "--sentence", "--max_gap", "1.0", "--max_line_width", "40", "--max_line_count", "1",
    ]},
    "parakeet": {"runner": ("asmr-parakeet", "parakeet_transcribe.py")},
    "whisperx": {"runner": ("asmr-whisperx", "whisperx_transcribe.py")},
}
DEFAULT_VARIANTS = ["base", "parallel2", "batched4", "batched8"]


def _ts(t: str) -> float:
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _spans(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for b in parse_srt_file(path):
        start, end = (x.strip().split()[0] for x in b.timestamp.split("-->"))
        out.append((_ts(start), _ts(end), "".join(b.text)))
    return out


def _norm(text: str) -> str:
    return "".join(c for c in text if c.isalnum())


_KAKASI = None


def to_kana(text: str) -> str | None:
    """Hiragana reading, so 雨/あめ/アメ compare equal. None if pykakasi is not installed."""
    global _KAKASI
    if _KAKASI is None:
        try:
            import pykakasi
        except ImportError:
            return None
        _KAKASI = pykakasi.kakasi()
    return "".join(item["hira"] for item in _KAKASI.convert(text))


def _edit_distance(a: str, b: str) -> int:
    # ponytail: O(len(a)*len(b)) pure Python; fine for a few files, use rapidfuzz if it gets slow
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def transcribe_external(runner: tuple[str, str], audio: Path, out_srt: Path) -> float:
    env, script = runner
    started = time.monotonic()
    subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", env, "python", str(PROJECT_ROOT / "scripts" / script), str(audio), str(out_srt)],
        check=True, shell=sys.platform == "win32",
    )
    return time.monotonic() - started


def transcribe(audio: Path, out_srt: Path, extra: list[str], config: lf.FasterWhisperConfig) -> tuple[float, float]:
    original = lf.build_faster_whisper_argv
    if extra:  # never patched for workers>1 variants, which have no extras
        lf.build_faster_whisper_argv = lambda *a: original(*a) + extra
    duration = [0.0]
    started = time.monotonic()
    try:
        lf.run_faster_whisper_transcription(
            audio_path=audio, transcribe_srt_path=out_srt, project_root=PROJECT_ROOT, config=config,
            transcription_progress_callback=lambda _r, _c, tot: duration.__setitem__(0, tot),
        )
    finally:
        lf.build_faster_whisper_argv = original
    return time.monotonic() - started, duration[0]


def metrics(ref: Path, cand: Path, audio_seconds: float) -> dict:
    r, c = _spans(ref), _spans(cand)
    rt, ct = _norm("".join(x[2] for x in r)), _norm("".join(x[2] for x in c))
    rk, ck = to_kana(rt), to_kana(ct)
    drifts = [min(abs(cs - rs) for rs, _, _ in r) for cs, _, _ in c] if r else []
    return {
        "blocks": len(c),
        "chars": len(ct),
        "cer": _edit_distance(rt, ct) / max(1, len(rt)),
        "kana_cer": None if rk is None else _edit_distance(rk, ck) / max(1, len(rk)),
        "repeats": sum(1 for a, b in zip(c, c[1:]) if a[2].strip() and a[2] == b[2]),
        "overrun": sum(1 for _, end, _ in c if audio_seconds and end > audio_seconds + 0.5),
        "drift": statistics.median(drifts) if drifts else 0.0,
    }


def save_meta(path: Path, files: list[dict], total: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"files": files, "total": total}), encoding="utf-8")


def load_meta(path: Path, file_count: int) -> dict | None:
    """{"files": [{"secs", "dur"}...], "total": wall seconds}, or None if missing/stale.
    An older plain-list format (files only) is accepted too."""
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(meta, list):
        meta = {"files": meta, "total": sum(f["secs"] for f in meta)}
    srts_ok = all((path.parent / f"{i:02d}.srt").exists() for i in range(file_count))
    return meta if len(meta["files"]) == file_count and srts_ok else None


def run_variant(name: str, files: list[Path], out_dir: Path, config: lf.FasterWhisperConfig, durations: list[float]) -> dict:
    spec = VARIANTS[name]
    srts = [out_dir / f"{i:02d}.srt" for i in range(len(files))]

    def one(i: int) -> dict:
        if "runner" in spec:
            return {"secs": transcribe_external(spec["runner"], files[i], srts[i]), "dur": durations[i]}
        secs, dur = transcribe(files[i], srts[i], spec.get("extra", []), config)
        return {"secs": secs, "dur": dur}

    started = time.monotonic()
    with ThreadPoolExecutor(spec.get("workers", 1)) as ex:
        results = list(ex.map(one, range(len(files))))
    meta = {"files": results, "total": time.monotonic() - started}
    save_meta(out_dir / "meta.json", results, meta["total"])
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Folder with test audio (searched recursively).")
    parser.add_argument("--max_files", type=int, default=5)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    parser.add_argument("--out", default=str(PROJECT_ROOT / ".asmr_bench_out"))
    parser.add_argument("--reuse_base", action="store_true", help="Skip re-running base; reuse its SRTs and timings from the last run.")
    parser.add_argument("--rescore", action="store_true", help="Transcribe nothing; recompute the table from SRTs already in --out.")
    args = parser.parse_args()

    files = sorted(p for p in Path(args.path).rglob("*") if p.suffix.lower() in AUDIO_EXTS)[: args.max_files]
    if not files:
        print(f"no audio files found under {args.path}", file=sys.stderr)
        return 2
    variants = ["base"] + [v for v in args.variants if v != "base"]
    config = lf.FasterWhisperConfig(
        model=DEFAULT_TRANSCRIBE_MODEL, language="ja", compute_type="float16",
        model_dir=lf.find_faster_whisper_model_dir(PROJECT_ROOT),
        initial_prompt=DEFAULT_ASMR_INITIAL_PROMPT, reprompt=True,
    )
    out_root = Path(args.out)
    if not args.rescore:
        lf.preflight_faster_whisper_model(PROJECT_ROOT, config.model)
    if to_kana("あ") is None:
        print("pykakasi not installed: kana CER shows n/a (pip install pykakasi)")
    print(f"{len(files)} files | variants: {' '.join(variants)} | SRTs in {out_root}\n")

    metas: dict[str, dict] = {}
    for v in variants:
        reuse = args.rescore or (v == "base" and args.reuse_base)
        if reuse:
            meta = load_meta(out_root / v / "meta.json", len(files))
            if meta is None and v == "base":
                print(f"{out_root / 'base'}: SRTs/meta.json missing or for a different file count; run base once first", file=sys.stderr)
                return 2
            if meta is None and not all((out_root / v / f"{i:02d}.srt").exists() for i in range(len(files))):
                print(f"  {v:<11} skipped: no SRTs in {out_root / v}")
                continue
            metas[v] = meta  # None = SRTs exist but no timing (scored for quality only)
        else:
            durations = [f["dur"] for f in metas["base"]["files"]] if "base" in metas and metas["base"] else [0.0] * len(files)
            metas[v] = run_variant(v, files, out_root / v, config, durations)
        print(f"  {v:<11} {'reused' if reuse else 'done':<6} " + (f"{metas[v]['total']:.0f}s" if metas[v] else "(no timing)"), flush=True)

    base_meta = metas["base"]
    durs = [f["dur"] for f in base_meta["files"]]
    print(f"\n{'variant':<12}{'time':>8}{'speedup':>9}{'xRT':>7}{'blocks':>8}{'chars':>8}{'CER':>8}{'kanaCER':>9}{'repeats':>9}{'overrun':>9}{'drift(s)':>10}")
    for v, meta in metas.items():
        ms = [metrics(out_root / "base" / f"{i:02d}.srt", out_root / v / f"{i:02d}.srt", durs[i]) for i in range(len(files))]
        kana = [m["kana_cer"] for m in ms]
        timing = (f"{meta['total']:>7.0f}s{base_meta['total'] / meta['total']:>8.2f}x{sum(durs) / meta['total']:>6.1f}x"
                  if meta else f"{'n/a':>8}{'n/a':>9}{'n/a':>7}")
        print(
            f"{v:<12}{timing}{sum(m['blocks'] for m in ms):>8}{sum(m['chars'] for m in ms):>8}"
            f"{statistics.mean(m['cer'] for m in ms):>7.1%}"
            + (f"{statistics.mean(kana):>8.1%}" if None not in kana else f"{'n/a':>9}")
            + f"{sum(m['repeats'] for m in ms):>9}{sum(m['overrun'] for m in ms):>9}"
            f"{statistics.median(m['drift'] for m in ms):>10.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
