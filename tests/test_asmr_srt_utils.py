from pathlib import Path

import pytest

from scripts.asmr_transcribe_lib.srt_utils import (
    SubtitleBlock,
    format_srt,
    parse_srt,
    replace_block_text,
    validate_same_timing,
)


def test_parse_and_format_simple_srt_roundtrip_preserves_index_timestamp_and_text():
    text = """1
00:00:01,000 --> 00:00:03,500
こんにちは。

2
00:00:04,000 --> 00:00:06,000
雨の音が聞こえる。
"""

    blocks = parse_srt(text)

    assert blocks == [
        SubtitleBlock(index="1", timestamp="00:00:01,000 --> 00:00:03,500", text=("こんにちは。",)),
        SubtitleBlock(index="2", timestamp="00:00:04,000 --> 00:00:06,000", text=("雨の音が聞こえる。",)),
    ]
    assert format_srt(blocks) == text


def test_parse_supports_utf8_bom_crlf_and_multiline_text():
    text = "\ufeff1\r\n00:00:01,000 --> 00:00:03,500\r\n一行目です。\r\n二行目です。\r\n\r\n2\r\n00:00:04,000 --> 00:00:06,000\r\n耳元で話しています。\r\n"

    blocks = parse_srt(text)

    assert blocks[0].index == "1"
    assert blocks[0].timestamp == "00:00:01,000 --> 00:00:03,500"
    assert blocks[0].text == ("一行目です。", "二行目です。")
    assert format_srt(blocks).startswith("1\n00:00:01,000 --> 00:00:03,500\n一行目です。\n二行目です。")


def test_parse_rejects_invalid_timestamp():
    text = """1
not-a-timestamp
こんにちは。
"""

    with pytest.raises(ValueError, match="timestamp"):
        parse_srt(text)


def test_replace_block_text_preserves_index_and_timestamp():
    blocks = [
        SubtitleBlock(index="7", timestamp="00:00:01,000 --> 00:00:03,500", text=("こんにちは。",)),
        SubtitleBlock(index="8", timestamp="00:00:04,000 --> 00:00:06,000", text=("雨の音。", "二行。")),
    ]

    replaced = replace_block_text(blocks, ["你好。", "雨聲。\n第二行。"])

    assert replaced == [
        SubtitleBlock(index="7", timestamp="00:00:01,000 --> 00:00:03,500", text=("你好。",)),
        SubtitleBlock(index="8", timestamp="00:00:04,000 --> 00:00:06,000", text=("雨聲。", "第二行。")),
    ]
    validate_same_timing(blocks, replaced)


def test_replace_block_text_rejects_count_mismatch():
    blocks = [SubtitleBlock(index="1", timestamp="00:00:01,000 --> 00:00:03,500", text=("こんにちは。",))]

    with pytest.raises(ValueError, match="block count"):
        replace_block_text(blocks, ["你好。", "多出的句子。"])


def test_validate_same_timing_rejects_timestamp_changes():
    original = [SubtitleBlock(index="1", timestamp="00:00:01,000 --> 00:00:03,500", text=("こんにちは。",))]
    changed = [SubtitleBlock(index="1", timestamp="00:00:02,000 --> 00:00:03,500", text=("你好。",))]

    with pytest.raises(ValueError, match="timestamp"):
        validate_same_timing(original, changed)


def test_parse_file_and_write_file_use_utf8(tmp_path: Path):
    from scripts.asmr_transcribe_lib.srt_utils import parse_srt_file, write_srt_file

    input_path = tmp_path / "input.srt"
    output_path = tmp_path / "output.srt"
    input_path.write_text("1\n00:00:01,000 --> 00:00:03,500\nこんにちは。\n", encoding="utf-8")

    blocks = parse_srt_file(input_path)
    write_srt_file(output_path, blocks)

    assert output_path.read_text(encoding="utf-8") == "1\n00:00:01,000 --> 00:00:03,500\nこんにちは。\n"
