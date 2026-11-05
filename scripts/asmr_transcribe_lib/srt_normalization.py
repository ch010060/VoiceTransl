"""Bounded display normalization, never acoustic alignment or a relaxed validator."""
from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence

from .srt_utils import SubtitleBlock, parse_srt, _validate_timestamp

MAX_BACKWARD_MS_PER_CUE = 1000
MAX_REVERSED_CUES = 10
MAX_REVERSED_PERCENT = 2
WARNING_CODE = "SRT_TIMESTAMP_NORMALIZED"
FALLBACK_WARNING_CODE = "SRT_TIMESTAMP_FALLBACK_USED"
FALLBACK_POLICY = "single-reversal-ratio-fallback-v1"


class TimestampOrderError(ValueError):
    def __init__(self, category: str):
        self.category = category
        super().__init__(f"SRT_TIMESTAMP_ORDER_INVALID:{category}")


def _milliseconds(value: str) -> int:
    h, m, tail = value.split(":")
    s, ms = tail.split(",")
    if int(m) >= 60 or int(s) >= 60:
        raise TimestampOrderError("timestamp_range")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def normalize_srt_start_order(blocks: Sequence[SubtitleBlock]) -> tuple[list[SubtitleBlock], list[dict]]:
    """Compare against the normalized predecessor; bound count, ratio and displacement.

    Ratio denominator is ALL parsed Cues, inclusive 2% (integer arithmetic).
    Only a single otherwise-valid reversal may bypass that ratio check.
    Total displacement is consequently bounded by 10 * 1000 ms per document.
    """
    if not blocks:
        raise TimestampOrderError("empty")
    result, changes = [], []
    previous = -1
    previous_text = ""
    for ordinal, block in enumerate(blocks, 1):
        if not block.index.isdigit():
            raise TimestampOrderError("index")
        try:
            _validate_timestamp(block.timestamp)
            start, end = block.timestamp.split(" --> ")
            left, right = _milliseconds(start), _milliseconds(end)
        except TimestampOrderError:
            raise
        except (ValueError, IndexError):
            raise TimestampOrderError("format") from None
        if right < left:
            raise TimestampOrderError("end_before_start")
        if left < previous:
            displacement = previous - left
            if displacement > MAX_BACKWARD_MS_PER_CUE:
                raise TimestampOrderError("displacement_limit")
            if previous > right:
                raise TimestampOrderError("repair_exceeds_end")
            timestamp = previous_text + " --> " + end
            changes.append({"cue": ordinal, "index": block.index, "original_timestamp": block.timestamp,
                            "normalized_timestamp": timestamp, "displacement_ms": displacement})
            block = replace(block, timestamp=timestamp)
            left, start = previous, previous_text
        result.append(block)
        previous, previous_text = left, start
    if len(changes) > MAX_REVERSED_CUES:
        raise TimestampOrderError("count_limit")
    if len(changes) * 100 > len(blocks) * MAX_REVERSED_PERCENT:
        if len(changes) != 1:
            raise TimestampOrderError("ratio_limit")
        changes[0]["fallback"] = {"fallback_used": True, "reversal_count": 1,
                                  "max_backward_ms": changes[0]["displacement_ms"],
                                  "normalization_policy": FALLBACK_POLICY}
    return result, changes


def normalize_srt_bytes(raw: bytes) -> tuple[bytes, list[dict]]:
    """Reuse the parser; edit only changed timestamp lines, keeping BOM/newlines/text."""
    try:
        content = raw.decode("utf-8")
        blocks = parse_srt(content)
    except (ValueError, UnicodeError):
        raise TimestampOrderError("format") from None
    normalized, changes = normalize_srt_start_order(blocks)
    if not changes:
        return raw, changes
    lines = content.splitlines(keepends=True)
    cue, position = 0, 0
    edits = {change["cue"]: change for change in changes}
    for offset, line in enumerate(lines):
        if not line.rstrip("\r\n"):
            position = 0
            continue
        position += 1
        if position == 2:
            cue += 1
            if cue in edits:
                change = edits[cue]
                lines[offset] = line.replace(change["original_timestamp"], change["normalized_timestamp"], 1)
    output = "".join(lines).encode("utf-8")
    if parse_srt(output.decode("utf-8")) != normalized:
        raise TimestampOrderError("serialization")
    return output, changes
