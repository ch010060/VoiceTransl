import pytest

from scripts.asmr_srt_diff_breakdown import breakdown
from scripts.asmr_transcribe_bench import to_kana

pytestmark = pytest.mark.skipif(to_kana("あ") is None, reason="pykakasi not installed")


def test_identical_and_spelling_only_differences_count_as_nothing():
    ref = [(0.0, 1.0, "雨の音"), (2.0, 3.0, "こんにちは")]
    cand = [(0.0, 1.0, "あめのおと"), (2.0, 3.0, "こんにちは")]

    b = breakdown(ref, cand)

    assert all(count == 0 for count, _ in b["edits"].values())
    assert b["dropped"] == (0, 0) and b["added"] == (0, 0)


def test_buckets_short_fillers_words_and_phrases():
    ref = [(0.0, 1.0, "きょうはいいてんきですね"), (2.0, 3.0, "こうえんにいきましょう")]
    # one inserted filler (1 kana), and the second line replaced by a different 9-kana phrase
    cand = [(0.0, 1.0, "きょうはあいいてんきですね"), (2.0, 3.0, "えきにかいものにいく")]

    b = breakdown(ref, cand)

    assert b["edits"]["1-2"][0] >= 1
    assert b["edits"]["7+"][0] >= 1


def test_dropped_and_added_lines_use_time_overlap():
    ref = [(0.0, 1.0, "あめ"), (5.0, 6.5, "かぜ")]
    cand = [(0.2, 0.9, "あめ"), (9.0, 10.0, "ゆき")]

    b = breakdown(ref, cand)

    assert b["dropped"] == (1, 1.5)  # the 5.0-6.5 s line has no candidate subtitle
    assert b["added"] == (1, 1.0)  # the 9-10 s line exists only in the candidate


def test_wrong_phrase_sharing_scattered_kana_counts_as_one_large_edit():
    from scripts.asmr_srt_diff_breakdown import _edit_sizes

    # differs everywhere except single shared kana (に, い) -> one merged 7+ edit, not many 1-2 edits
    assert _edit_sizes("こうえんにいきましょう", "えきにかいものにいく") == [11]
    assert _edit_sizes("あいうえお", "あいうえお") == []
    # two fillers far apart stay two small edits
    assert _edit_sizes("きょうはいいてんきですねあしたもはれるかな", "きょうはあいいてんきですねあしたもはれるかなあ") == [1, 1]
