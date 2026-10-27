"""A/B benchmark for the Sakura translation step. Prints counts and timings only.

Usage:
    python scripts/asmr_translate_bench.py --path W:\\bench_in [--repeat 2]

Runs every variant on the same *-transcribe.srt files (up to --max_files, in parallel
like the batch CLI) and writes translated copies to a temp folder, never next to audio.
"""
from __future__ import annotations

import argparse
import collections
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asmr_transcribe_lib import openai_compatible as oc  # noqa: E402
from asmr_transcribe_lib import translation as tr  # noqa: E402
from asmr_transcribe_lib.srt_utils import parse_srt_file  # noqa: E402

API = "http://127.0.0.1:8080/v1"
MODEL = "sakura-14b-qwen2.5-v1.0-q4km.gguf"

# name -> (lines per request, retry the same chunk on a line-count mismatch?)
VARIANTS = {
    "A current (30, retry)": (30, True),
    "B 30, split on mismatch": (30, False),
    "C 20, retry": (20, True),
    "D 20, split on mismatch": (20, False),
    "E 15, split on mismatch": (15, False),
}

_original_chat = tr.chat_completion_json_array
_original_retry = tr._translate_chunk_with_retry


def _is_count_mismatch(exc: Exception) -> bool:
    return "count mismatch" in str(exc)


def run_variant(name: str, chunk: int, retry_mismatch: bool, files: list[Path], out_dir: Path, workers: int) -> dict:
    stats = collections.Counter()
    lock = __import__("threading").Lock()

    def counted_chat(**kwargs):
        with lock:
            stats["requests"] += 1
        try:
            return _original_chat(**kwargs)
        except oc.OpenAICompatibleError as exc:
            with lock:
                stats["failed"] += 1
                stats["mismatch" if _is_count_mismatch(exc) else "other_error"] += 1
            raise

    def retry(texts, *, base_url, model, client, max_retries):
        if retry_mismatch:
            return _original_retry(texts, base_url=base_url, model=model, client=client, max_retries=max_retries)
        last = None
        for attempt in range(max_retries):
            try:
                return tr.chat_completion_json_array(
                    base_url=base_url, model=model, texts=texts, client=client,
                    frequency_penalty=oc.DEFAULT_FREQUENCY_PENALTY + 0.1 * attempt,
                )
            except oc.OpenAICompatibleError as exc:
                last = exc
                if _is_count_mismatch(exc):
                    break  # same chunk usually drifts again; split instead
        raise last

    tr.chat_completion_json_array = counted_chat
    tr._translate_chunk_with_retry = retry

    def one(src: Path) -> tuple[bool, int, int]:
        dst = out_dir / f"{src.stem}-{chunk}-{int(retry_mismatch)}.srt"
        try:
            tr.translate_srt_to_simplified(src, dst, base_url=API, model=MODEL, max_blocks=chunk)
        except Exception:
            return False, 0, 0
        a, b = parse_srt_file(src), parse_srt_file(dst)
        # quality proxy: output lines that are empty or still contain kana
        bad = sum(
            1 for y in b
            if not "".join(y.text).strip() or any("\u3040" <= c <= "\u30ff" for c in "".join(y.text))
        )
        return len(a) == len(b), len(b), bad

    started = time.monotonic()
    try:
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(one, files))
    finally:
        tr.chat_completion_json_array = _original_chat
        tr._translate_chunk_with_retry = _original_retry
    return {
        "name": name,
        "seconds": time.monotonic() - started,
        "files_ok": sum(1 for ok, _, _ in results if ok),
        "lines": sum(n for _, n, _ in results),
        "bad_lines": sum(bad for _, _, bad in results),
        **stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Folder with *-transcribe.srt files (searched recursively).")
    parser.add_argument("--max_files", type=int, default=5)
    parser.add_argument("--workers", type=int, default=5, help="Concurrent files, like --parallel in the batch CLI.")
    parser.add_argument("--repeat", type=int, default=1, help="Run each variant N times; Sakura sampling is not fully deterministic.")
    parser.add_argument("--only", nargs="*", help="Variant letters to run, e.g. --only A B")
    args = parser.parse_args()

    files = sorted(Path(args.path).rglob("*-transcribe.srt"))[: args.max_files]
    if not files:
        print(f"no *-transcribe.srt files found under {args.path}", file=sys.stderr)
        return 2
    oc.preflight_model(API, MODEL)
    total_blocks = sum(len(parse_srt_file(f)) for f in files)
    print(f"{len(files)} files, {total_blocks} subtitle lines, workers={args.workers}, repeat={args.repeat}\n")
    print(f"{'variant':<26}{'run':>4}{'requests':>10}{'failed':>8}{'fail%':>7}{'mismatch':>10}{'other':>7}{'time':>8}{'files ok':>10}{'bad lines':>11}")

    variants = {k: v for k, v in VARIANTS.items() if not args.only or k[0] in args.only}
    with tempfile.TemporaryDirectory(prefix="asmr_bench_") as tmp:
        for run in range(1, args.repeat + 1):
            for name, (chunk, retry_mismatch) in variants.items():
                r = run_variant(name, chunk, retry_mismatch, files, Path(tmp), args.workers)
                pct = 100 * r.get("failed", 0) / max(1, r.get("requests", 0))
                print(
                    f"{r['name']:<26}{run:>4}{r.get('requests', 0):>10}{r.get('failed', 0):>8}{pct:>6.0f}%"
                    f"{r.get('mismatch', 0):>10}{r.get('other_error', 0):>7}{r['seconds']:>7.0f}s"
                    f"{r['files_ok']:>6}/{len(files):<3}{r['bad_lines']:>11}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
