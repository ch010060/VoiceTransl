from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from collections.abc import Sequence

_TIMESTAMP_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}(?:\s+.*)?$"
)


@dataclass(frozen=True)
class SubtitleBlock:
    index: str
    timestamp: str
    text: tuple[str, ...]


def parse_srt_file(path: Path | str) -> list[SubtitleBlock]:
    return parse_srt(Path(path).read_text(encoding="utf-8-sig"))


def write_srt_file(path: Path | str, blocks: Sequence[SubtitleBlock]) -> None:
    Path(path).write_text(format_srt(blocks), encoding="utf-8")


def parse_srt(content: str) -> list[SubtitleBlock]:
    normalized = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    blocks: list[SubtitleBlock] = []
    current: list[str] = []

    for line in lines:
        if line == "":
            if current:
                blocks.append(_parse_block(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(_parse_block(current))

    return blocks


def format_srt(blocks: Sequence[SubtitleBlock]) -> str:
    rendered: list[str] = []
    for block in blocks:
        _validate_timestamp(block.timestamp)
        rendered.append(str(block.index))
        rendered.append(block.timestamp)
        rendered.extend(block.text)
        rendered.append("")
    return "\n".join(rendered)


def replace_block_text(blocks: Sequence[SubtitleBlock], replacement_texts: Sequence[str]) -> list[SubtitleBlock]:
    if len(blocks) != len(replacement_texts):
        raise ValueError(f"subtitle block count mismatch: {len(blocks)} blocks vs {len(replacement_texts)} replacements")

    replaced: list[SubtitleBlock] = []
    for block, replacement in zip(blocks, replacement_texts, strict=True):
        lines = tuple(str(replacement).replace("\r\n", "\n").replace("\r", "\n").split("\n"))
        replaced.append(SubtitleBlock(index=block.index, timestamp=block.timestamp, text=lines))
    return replaced


def validate_same_timing(original: Sequence[SubtitleBlock], candidate: Sequence[SubtitleBlock]) -> None:
    if len(original) != len(candidate):
        raise ValueError(f"subtitle block count mismatch: {len(original)} original vs {len(candidate)} candidate")
    for offset, (left, right) in enumerate(zip(original, candidate, strict=True), start=1):
        if left.index != right.index:
            raise ValueError(f"subtitle index changed at block {offset}: {left.index!r} != {right.index!r}")
        if left.timestamp != right.timestamp:
            raise ValueError(f"subtitle timestamp changed at block {offset}: {left.timestamp!r} != {right.timestamp!r}")


def _parse_block(lines: Sequence[str]) -> SubtitleBlock:
    if len(lines) < 2:
        raise ValueError(f"invalid SRT block: expected index and timestamp, got {list(lines)!r}")
    index = lines[0].strip()
    timestamp = lines[1].strip()
    _validate_timestamp(timestamp)
    return SubtitleBlock(index=index, timestamp=timestamp, text=tuple(lines[2:]))


def _validate_timestamp(timestamp: str) -> None:
    if not _TIMESTAMP_RE.match(timestamp):
        raise ValueError(f"invalid SRT timestamp: {timestamp!r}")
