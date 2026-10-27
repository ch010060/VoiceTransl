from pathlib import Path

import pytest

from scripts.asmr_transcribe_lib.opencc_utils import convert_blocks_opencc, convert_srt_file_opencc
from scripts.asmr_transcribe_lib.srt_utils import SubtitleBlock, parse_srt_file


def test_convert_blocks_opencc_converts_simplified_text_to_traditional_only():
    blocks = [
        SubtitleBlock(index="1", timestamp="00:00:01,000 --> 00:00:02,000", text=("汉字。",)),
        SubtitleBlock(index="2", timestamp="00:00:03,000 --> 00:00:04,000", text=("能听见雨声。", "第二行文本。")),
    ]

    converted = convert_blocks_opencc(blocks, config="s2twp")

    assert converted == [
        SubtitleBlock(index="1", timestamp="00:00:01,000 --> 00:00:02,000", text=("漢字。",)),
        SubtitleBlock(index="2", timestamp="00:00:03,000 --> 00:00:04,000", text=("能聽見雨聲。", "第二行文本。")),
    ]


def test_convert_srt_file_opencc_writes_opencc_srt_and_preserves_timing(tmp_path: Path):
    source = tmp_path / "track-translate.srt"
    target = tmp_path / "track-opencc.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n汉字。\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n能听见雨声。\n",
        encoding="utf-8",
    )

    result = convert_srt_file_opencc(source, target, config="s2twp")

    assert result == target
    converted = parse_srt_file(target)
    assert [block.index for block in converted] == ["1", "2"]
    assert [block.timestamp for block in converted] == [
        "00:00:01,000 --> 00:00:02,000",
        "00:00:03,000 --> 00:00:04,000",
    ]
    assert converted[0].text == ("漢字。",)
    assert converted[1].text == ("能聽見雨聲。",)


def test_convert_srt_file_opencc_removes_stale_target_on_failure(tmp_path: Path):
    source = tmp_path / "bad-translate.srt"
    target = tmp_path / "bad-opencc.srt"
    source.write_text("not a valid srt", encoding="utf-8")
    target.write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError):
        convert_srt_file_opencc(source, target, config="s2twp")

    assert not target.exists()
