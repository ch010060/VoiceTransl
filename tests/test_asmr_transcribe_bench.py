from pathlib import Path

from scripts.asmr_transcribe_bench import VARIANTS, _edit_distance, load_meta, metrics, save_meta, to_kana
from scripts.asmr_transcribe_lib.cli import DEFAULT_ASMR_INITIAL_PROMPT
from scripts.whisperx_transcribe import ASMR_PROMPT, segments_to_srt


def write_srt(path: Path, blocks):
    path.write_text(
        "\n".join(f"{i}\n{s} --> {e}\n{t}\n" for i, (s, e, t) in enumerate(blocks, start=1)),
        encoding="utf-8",
    )


def test_meta_round_trip_and_rejects_mismatch(tmp_path: Path):
    meta = tmp_path / "base" / "meta.json"
    files = [{"secs": 45.6, "dur": 175.0}, {"secs": 350.2, "dur": 1320.0}]
    save_meta(meta, files, total=395.8)
    for i in range(2):
        write_srt(meta.parent / f"{i:02d}.srt", [("00:00:01,000", "00:00:02,000", "あ")])

    assert load_meta(meta, 2) == {"files": files, "total": 395.8}
    assert load_meta(meta, 3) is None  # different file count
    (meta.parent / "01.srt").unlink()
    assert load_meta(meta, 2) is None  # a SRT is missing
    assert load_meta(tmp_path / "missing.json", 1) is None


def test_load_meta_accepts_old_list_format(tmp_path: Path):
    meta = tmp_path / "meta.json"
    meta.write_text('[{"secs": 1.5, "dur": 10.0}, {"secs": 2.5, "dur": 20.0}]', encoding="utf-8")
    for i in range(2):
        write_srt(tmp_path / f"{i:02d}.srt", [("00:00:01,000", "00:00:02,000", "あ")])

    assert load_meta(meta, 2)["total"] == 4.0


def test_metrics_counts_repeats_overrun_and_cer(tmp_path: Path):
    ref, cand = tmp_path / "ref.srt", tmp_path / "cand.srt"
    write_srt(ref, [("00:00:01,000", "00:00:02,000", "こんにちは。"), ("00:00:03,000", "00:00:04,000", "雨の音。")])
    write_srt(cand, [
        ("00:00:01,200", "00:00:02,000", "こんにちわ"),
        ("00:00:03,000", "00:00:09,000", "雨の音"),
        ("00:00:09,000", "00:00:09,500", "雨の音"),
    ])

    m = metrics(ref, cand, audio_seconds=8.0)

    assert (m["blocks"], m["repeats"], m["overrun"]) == (3, 1, 2)
    assert round(m["drift"], 2) == 0.2
    assert _edit_distance("kitten", "sitting") == 3


def test_kana_cer_ignores_kanji_versus_kana_spelling(tmp_path: Path):
    if to_kana("あ") is None:
        import pytest
        pytest.skip("pykakasi not installed")
    ref, cand = tmp_path / "ref.srt", tmp_path / "cand.srt"
    write_srt(ref, [("00:00:01,000", "00:00:02,000", "雨の音")])
    write_srt(cand, [("00:00:01,000", "00:00:02,000", "あめのおと")])

    m = metrics(ref, cand, audio_seconds=5.0)

    assert m["cer"] > 0.5  # spelled differently
    assert m["kana_cer"] == 0.0  # but the same reading


def test_variant_specs_are_well_formed():
    for name, spec in VARIANTS.items():
        assert set(spec) <= {"extra", "workers", "runner"}, name
        assert not (spec.get("workers", 1) > 1 and spec.get("extra")), f"{name}: parallel variants must not patch argv"
    assert VARIANTS["batched8"]["extra"] == ["--batched", "--batch_size", "8"]
    assert "--unmerged" not in VARIANTS["batched4"]["extra"]


def test_whisperx_srt_drops_unaligned_and_empty_segments():
    srt = segments_to_srt([
        {"start": 0.0, "end": 1.25, "text": " こんにちは "},
        {"text": "時間なし"},  # alignment failed
        {"start": 2.0, "end": 3.0, "text": "  "},
        {"start": 3.0, "end": 4.5, "text": "雨の音"},
    ])

    assert srt == "1\n00:00:00,000 --> 00:00:01,250\nこんにちは\n\n2\n00:00:03,000 --> 00:00:04,500\n雨の音\n"


def test_whisperx_prompt_matches_production_prompt():
    assert ASMR_PROMPT == DEFAULT_ASMR_INITIAL_PROMPT
