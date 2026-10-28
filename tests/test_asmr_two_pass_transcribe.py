from scripts.asmr_two_pass_transcribe import select_windows, splice_segments, segments_to_srt


def segment(start, end, text="テスト", **scores):
    return dict(start=start, end=end, text=text, avg_logprob=-0.2,
                no_speech_prob=0.01, compression_ratio=1.0) | scores


def test_selection_uses_each_confidence_signal_and_clamps_to_audio():
    items = [segment(0, 2, avg_logprob=-1.1), segment(10, 12, no_speech_prob=0.8),
             segment(20, 22, compression_ratio=3), segment(30, 32)]
    assert select_windows(items, 32) == [(0, 3), (9, 13), (19, 23)]
    assert select_windows([segment(30, 32, avg_logprob=-1.1)], 32) == [(29, 32)]


def test_selection_expands_whole_intersecting_captions_and_merges():
    items = [segment(1, 4), segment(4.5, 6, avg_logprob=-1.1),
             segment(6.5, 8, compression_ratio=3), segment(8.5, 10), segment(11, 12)]
    assert select_windows(items, 15) == [(1, 10)]


def test_selection_leaves_healthy_segments_and_silence_alone():
    assert select_windows([segment(5, 7)], 20) == []
    assert select_windows([], 20) == []


def test_splice_preserves_unselected_text_offsets_clips_and_sorts_retry():
    original = [segment(0, 2, "前"), segment(3, 5, "誤り"), segment(8, 10, "後")]
    retried = [segment(1, 2, "訂正"), segment(-1, 0.5, "最初"), segment(3, 9, "最後")]
    result = splice_segments(original, [(3, 6, retried)])
    assert [(s["start"], s["end"], s["text"]) for s in result] == [
        (0, 2, "前"), (3, 3.5, "最初"), (4, 5, "訂正"), (8, 10, "後")]


def test_empty_retry_retains_original_instead_of_silently_deleting_speech():
    original = [segment(3, 5, "元の字幕")]
    assert splice_segments(original, [(2, 6, [])]) == original
    assert splice_segments(original, [(2, 6, [segment(0, 2, " ")])]) == original


def test_srt_has_valid_millisecond_rounding_and_skips_empty_spans():
    assert segments_to_srt([segment(59.9996, 61.25, "こんにちは。"),
                            segment(4, 4, "無効"), segment(5, 6, " ")]) == (
        "1\n00:01:00,000 --> 00:01:01,250\nこんにちは。\n")


def test_srt_splits_at_word_punctuation_using_word_timestamps():
    item = segment(0, 5, "こんにちは。元気？", words=[
        dict(start=0.5, end=2, word="こんにちは。"),
        dict(start=3, end=4.5, word="元気？")])
    assert segments_to_srt([item]) == (
        "1\n00:00:00,500 --> 00:00:02,000\nこんにちは。\n\n"
        "2\n00:00:03,000 --> 00:00:04,500\n元気？\n")
