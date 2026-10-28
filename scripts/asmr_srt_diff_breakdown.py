"""Classify the differences between a reference SRT and a candidate SRT. Prints counts only.

Usage:
    python scripts/asmr_srt_diff_breakdown.py REF.srt CAND.srt [CAND2.srt ...]

Both sides are compared as kana (so 雨 and あめ count as equal). Reported:
  dropped lines   reference subtitles with no candidate subtitle overlapping them in time
  added lines     candidate subtitles with no reference subtitle overlapping them (possible hallucination)
  text edits      each differing span, bucketed by size: 1-2 kana (fillers, breaths, short sounds),
                  3-6 kana (a word), 7+ kana (a phrase or sentence)
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asmr_transcribe_bench import _norm, _spans, to_kana  # noqa: E402

BUCKETS = (("1-2", 1, 2), ("3-6", 3, 6), ("7+", 7, 10**9))


def _overlaps(span: tuple[float, float, str], others: list[tuple[float, float, str]]) -> bool:
    start, end, _ = span
    return any(s < end and start < e for s, e, _ in others)


def _edit_sizes(ref: str, cand: str, *, bridge: int = 2) -> list[int]:
    """Size of each differing span. difflib splits a wrong phrase at every kana that happens to
    match (に, い, ...), so edits separated by <= `bridge` equal kana are merged into one."""
    sizes: list[int] = []
    group: list[int] | None = None  # [ref chars, cand chars] of the current merged edit
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, cand, autojunk=False).get_opcodes():
        if op == "equal":
            if group is not None and i2 - i1 <= bridge:
                group[0] += i2 - i1
                group[1] += j2 - j1
            elif group is not None:
                sizes.append(max(group))
                group = None
            continue
        if group is None:
            group = [0, 0]
        group[0] += i2 - i1
        group[1] += j2 - j1
    if group is not None:
        sizes.append(max(group))
    return sizes


def breakdown(ref: list[tuple[float, float, str]], cand: list[tuple[float, float, str]]) -> dict:
    kana = lambda spans: to_kana(_norm("".join(t for _, _, t in spans))) or ""
    rk, ck = kana(ref), kana(cand)
    edits = {name: [0, 0] for name, _, _ in BUCKETS}  # bucket -> [count, kana chars changed]
    for size in _edit_sizes(rk, ck):
        name = next(n for n, lo, hi in BUCKETS if lo <= size <= hi)
        edits[name][0] += 1
        edits[name][1] += size
    dropped = [r for r in ref if r[2].strip() and not _overlaps(r, cand)]
    added = [c for c in cand if c[2].strip() and not _overlaps(c, ref)]
    return {
        "ref_kana": len(rk),
        "edits": edits,
        "dropped": (len(dropped), sum(e - s for s, e, _ in dropped)),
        "added": (len(added), sum(e - s for s, e, _ in added)),
    }


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    if to_kana("あ") is None:
        print("pykakasi is required: pip install pykakasi", file=sys.stderr)
        return 2
    ref = _spans(Path(sys.argv[1]))
    for cand_path in sys.argv[2:]:
        b = breakdown(ref, _spans(Path(cand_path)))
        changed = sum(chars for _, chars in b["edits"].values())
        print(f"\n{cand_path}")
        print(f"  reference: {len(ref)} lines, {b['ref_kana']} kana")
        print(f"  dropped lines: {b['dropped'][0]} ({b['dropped'][1]:.1f}s of reference speech with no subtitle)")
        print(f"  added lines:   {b['added'][0]} ({b['added'][1]:.1f}s where only the candidate has a subtitle)")
        print(f"  text edits by size (kana):  {'count':>6} {'kana':>6} {'share of all changed kana':>27}")
        for name, (count, chars) in b["edits"].items():
            print(f"    {name:<26}{count:>6} {chars:>6} {chars / max(1, changed):>26.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
