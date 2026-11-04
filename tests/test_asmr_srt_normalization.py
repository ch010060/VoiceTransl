"""Synthetic Gate A timestamps; no private captions or audio fixtures."""
from dataclasses import replace

import pytest

from scripts.asmr_transcribe_lib.srt_normalization import (
    TimestampOrderError, normalize_srt_bytes, normalize_srt_start_order,
)
from scripts.asmr_transcribe_lib.srt_utils import SubtitleBlock, format_srt, parse_srt


def ts(ms):
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def fixture(count=175, reversals=None):
    starts = [i * 9800 for i in range(count)]
    starts[74:76] = [725460, 725040]
    blocks = [SubtitleBlock(str(i + 1), ts(start) + " --> " + ts(start + 9000),
                            (f"Synthetic {i + 1}", "second line")) for i, start in enumerate(starts)]
    blocks[74] = replace(blocks[74], timestamp="00:12:05,460 --> 00:12:05,560")
    blocks[75] = replace(blocks[75], timestamp="00:12:05,040 --> 00:12:19,000")
    if reversals is not None:
        blocks = [SubtitleBlock(str(i + 1), ts(i * 10000) + " --> " + ts(i * 10000 + 9000),
                                (f"Synthetic {i + 1}",)) for i in range(count)]
        for offset, displacement in reversals.items():
            previous = offset * 10000 - 10000
            blocks[offset] = replace(blocks[offset], timestamp=ts(previous - displacement) + " --> " + ts(previous + 9000))
    return blocks


def test_real_rca_minimum_change():
    original = fixture()
    normalized, changes = normalize_srt_start_order(original)
    assert len(changes) == 1 and changes[0]["cue"] == 76
    assert changes[0]["displacement_ms"] == 420
    assert normalized[75].timestamp == "00:12:05,460 --> 00:12:19,000"
    assert original[75].timestamp == "00:12:05,040 --> 00:12:19,000"
    assert [(b.index, b.text) for b in normalized] == [(b.index, b.text) for b in original]
    assert normalized[:75] == original[:75] and normalized[76:] == original[76:]


@pytest.mark.parametrize("bom,newline", [(b"", "\n"), (b"\xef\xbb\xbf", "\r\n"), (b"", "\r")])
def test_surgical_bytes_and_noop(bom, newline):
    raw = bom + format_srt(fixture()).replace("\n", newline).encode()
    fixed, changes = normalize_srt_bytes(raw)
    assert fixed == raw.replace(b"00:12:05,040 -->", b"00:12:05,460 -->")
    assert len(changes) == 1
    assert normalize_srt_bytes(fixed) == (fixed, [])


@pytest.mark.parametrize("count,offsets,passes", [
    (175, {10: 420}, True), (100, {10: 1, 20: 2}, True),
    (99, {10: 1, 20: 2}, False), (175, {10: 1, 20: 2, 30: 3}, True),
    (175, {10: 1, 20: 2, 30: 3, 40: 4}, False),
    (550, {i * 10: 10 for i in range(1, 11)}, True),
    (550, {i * 10: 10 for i in range(1, 12)}, False),
])
def test_count_ratio_inclusive(count, offsets, passes):
    blocks = fixture(count, offsets)
    if passes:
        assert len(normalize_srt_start_order(blocks)[1]) == len(offsets)
    else:
        with pytest.raises(TimestampOrderError, match="ratio_limit|count_limit"):
            normalize_srt_start_order(blocks)


@pytest.mark.parametrize("displacement,passes", [(999, True), (1000, True), (1001, False), (5000, False)])
def test_displacement(displacement, passes):
    blocks = fixture(reversals={10: displacement})
    if passes:
        assert normalize_srt_start_order(blocks)[1][0]["displacement_ms"] == displacement
    else:
        with pytest.raises(TimestampOrderError, match="displacement_limit"):
            normalize_srt_start_order(blocks)


def test_consecutive_chain_measured_from_normalized_start():
    blocks = fixture(reversals={})
    for offset, start in [(10, 89500), (11, 89000)]:
        blocks[offset] = replace(blocks[offset], timestamp=ts(start) + " --> " + ts(120000))
    fixed, changes = normalize_srt_start_order(blocks)
    assert [c["displacement_ms"] for c in changes] == [500, 1000]
    assert [b.timestamp.split(" --> ")[0] for b in fixed[9:12]] == [ts(90000)] * 3
    blocks[11] = replace(blocks[11], timestamp=ts(88999) + " --> " + ts(120000))
    with pytest.raises(TimestampOrderError, match="displacement_limit"):
        normalize_srt_start_order(blocks)


@pytest.mark.parametrize("timestamp,error", [
    ("00:12:05,040 --> 00:12:05,450", "repair_exceeds_end"),
    ("00:12:05,040 --> 00:12:05,030", "end_before_start"),
    ("00:60:05,040 --> 01:12:19,000", "timestamp_range"),
    ("-1:12:05,040 --> 00:12:19,000", "format"),
])
def test_reject_without_altering_end(timestamp, error):
    blocks = fixture()
    blocks[75] = replace(blocks[75], timestamp=timestamp)
    raw = "\n\n".join(f"{b.index}\n{b.timestamp}\nSynthetic" for b in blocks).encode()
    with pytest.raises(TimestampOrderError, match=error):
        normalize_srt_bytes(raw)


@pytest.mark.parametrize("raw", [b"", b"WEBVTT", b"1\nno timestamp\ntext", b"\xff",
                                  b"bad\n00:00:00,000 --> 00:00:01,000\ntext"])
def test_malformed(raw):
    with pytest.raises(TimestampOrderError):
        normalize_srt_bytes(raw)


def test_legal_overlap_zero_duration_and_long_gaps_unchanged():
    raw = b"1\n00:00:00,000 --> 00:00:09,000\na\n\n2\n00:00:01,000 --> 00:00:01,000\nb\n\n3\n01:00:00,000 --> 01:00:01,000\nc\n"
    assert normalize_srt_bytes(raw) == (raw, [])
    assert len(parse_srt(raw.decode())) == 3
