from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import opencc

from .srt_utils import SubtitleBlock, parse_srt_file, validate_same_timing, write_srt_file


def convert_blocks_opencc(blocks: Sequence[SubtitleBlock], *, config: str = "s2twp") -> list[SubtitleBlock]:
    converter = opencc.OpenCC(config)
    converted: list[SubtitleBlock] = []
    for block in blocks:
        converted.append(
            SubtitleBlock(
                index=block.index,
                timestamp=block.timestamp,
                text=tuple(converter.convert(line) for line in block.text),
            )
        )
    validate_same_timing(blocks, converted)
    return converted


def convert_srt_file_opencc(
    source_srt_path: Path | str,
    target_srt_path: Path | str,
    *,
    config: str = "s2twp",
) -> Path:
    source = Path(source_srt_path)
    target = Path(target_srt_path)
    tmp_target = target.with_name(f"{target.name}.tmp")

    if target.exists():
        target.unlink()
    if tmp_target.exists():
        tmp_target.unlink()

    try:
        original = parse_srt_file(source)
        converted = convert_blocks_opencc(original, config=config)
        validate_same_timing(original, converted)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_srt_file(tmp_target, converted)
        tmp_target.replace(target)
    except Exception:
        if tmp_target.exists():
            tmp_target.unlink()
        if target.exists():
            target.unlink()
        raise
    return target
