from scripts.parakeet_transcribe import _ts, split_chars


def c(ch, start, end):
    # NeMo 3.0 shape: each char is a list of strings
    return {"char": [ch], "start": start, "end": end}


def test_split_chars_breaks_on_sentence_end_and_pause():
    chars = [c("あ", 0.0, 0.1), c("め", 0.1, 0.2), c("。", 0.2, 0.25), c("か", 0.3, 0.4), c("ぜ", 0.4, 0.5), c("ね", 1.5, 1.6)]

    assert split_chars(chars) == [(0.0, 0.25, "あめ。"), (0.3, 0.5, "かぜ"), (1.5, 1.6, "ね")]


def test_split_chars_caps_length_and_duration():
    chars = [c("x", i * 0.1, i * 0.1 + 0.05) for i in range(100)]

    segments = split_chars(chars, max_chars=40, max_seconds=8.0)

    assert all(len(text) <= 40 for _, _, text in segments)
    assert all(end - start <= 8.0 for start, end, _ in segments)
    assert "".join(text for _, _, text in segments) == "x" * 100


def test_split_chars_accepts_plain_strings_and_drops_empty_chars():
    chars = [{"char": "", "start": 0.0, "end": 0.2}, c("", 0.2, 0.3), {"char": "心", "start": 0.3, "end": 0.5}]

    assert split_chars(chars) == [(0.0, 0.5, "心")]
    assert split_chars([]) == []


def test_ts_formats_srt_timestamp():
    assert _ts(0) == "00:00:00,000"
    assert _ts(3725.0456) == "01:02:05,046"
