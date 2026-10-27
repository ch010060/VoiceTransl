from pathlib import Path

import pytest

from scripts.asmr_transcribe_lib.planner import discover_audio_tasks


def touch(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_discovers_first_level_works_audio_types_and_stable_order(tmp_path: Path):
    root = tmp_path / "root"
    touch(root / "work_b" / "02" / "b.wav")
    touch(root / "work_a 日本語" / "01" / "a.flac")
    touch(root / "work_a 日本語" / "02" / "c.MP3")
    touch(root / "work_a 日本語" / "ignore" / "cover.jpg")
    touch(root / "root_audio.wav")

    tasks = discover_audio_tasks(root)

    assert [task.work_name for task in tasks] == [
        "work_a 日本語",
        "work_a 日本語",
        "work_b",
        "_root",
    ]
    assert [task.relative_audio_path.as_posix() for task in tasks] == [
        "work_a 日本語/01/a.flac",
        "work_a 日本語/02/c.MP3",
        "work_b/02/b.wav",
        "root_audio.wav",
    ]
    assert all(task.status == "queued" for task in tasks)


def test_marks_existing_srt_as_skipped_without_overwrite(tmp_path: Path):
    root = tmp_path / "root"
    audio = touch(root / "work" / "01" / "track.flac")
    final_srt = touch(root / "work" / "01" / "track.srt", "1\n00:00:00,000 --> 00:00:01,000\n既有字幕\n")

    [task] = discover_audio_tasks(root)

    assert task.audio_path == audio
    assert task.final_srt_path == final_srt
    assert task.status == "skipped"
    assert task.action == "skip"
    assert task.reason == "existing_srt"


def test_marks_existing_alternate_subtitle_for_copy_not_transcription(tmp_path: Path):
    root = tmp_path / "root"
    audio = touch(root / "work" / "bonus" / "track.mp3")
    vtt = touch(root / "work" / "bonus" / "track.vtt", "WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n")

    [task] = discover_audio_tasks(root)

    assert task.audio_path == audio
    assert task.status == "skipped"
    assert task.action == "copy_existing_subtitle"
    assert task.reason == "existing_alt_subtitle:.vtt"
    assert task.existing_subtitle_path == vtt
    assert task.final_srt_path == audio.with_suffix(".srt")


def test_same_work_duplicate_audio_stem_processes_best_source_once_and_copies_result(tmp_path: Path):
    root = tmp_path / "root"
    mp3 = touch(root / "work" / "mp3" / "Track01.mp3")
    flac = touch(root / "work" / "flac" / "Track01.flac")

    tasks = discover_audio_tasks(root)

    by_audio = {task.audio_path: task for task in tasks}
    assert by_audio[flac].status == "queued"
    assert by_audio[flac].action == "process"
    assert by_audio[mp3].status == "skipped"
    assert by_audio[mp3].action == "copy_duplicate_audio_subtitle"
    assert by_audio[mp3].existing_subtitle_path == flac.with_suffix(".srt")


def test_same_work_duplicate_audio_stem_reuses_existing_srt_from_other_directory(tmp_path: Path):
    root = tmp_path / "root"
    existing_audio = touch(root / "work" / "flac" / "Track01.flac")
    existing_srt = touch(root / "work" / "flac" / "Track01.srt", "existing subtitle")
    mp3 = touch(root / "work" / "mp3" / "Track01.mp3")

    tasks = discover_audio_tasks(root)

    by_audio = {task.audio_path: task for task in tasks}
    assert by_audio[existing_audio].status == "skipped"
    assert by_audio[existing_audio].action == "skip"
    assert by_audio[mp3].status == "skipped"
    assert by_audio[mp3].action == "copy_existing_subtitle"
    assert by_audio[mp3].reason == "existing_duplicate_subtitle:.srt"
    assert by_audio[mp3].existing_subtitle_path == existing_srt


def test_same_work_se_nashi_audio_is_treated_as_duplicate_content(tmp_path: Path):
    root = tmp_path / "root"
    se_ari = touch(root / "work" / "01_SEあり" / "Track01.flac")
    se_nashi_suffix = touch(root / "work" / "02_SEなし" / "Track01_SEなし.wav")
    se_nashi_prefix = touch(root / "work" / "03_SEなし" / "【SEなし】Track01.mp3")
    se_ari.with_suffix(".srt").write_text("existing subtitle", encoding="utf-8")

    tasks = discover_audio_tasks(root)

    by_audio = {task.audio_path: task for task in tasks}
    assert by_audio[se_ari].action == "skip"
    assert by_audio[se_nashi_suffix].action == "copy_existing_subtitle"
    assert by_audio[se_nashi_suffix].reason == "existing_duplicate_subtitle:.srt"
    assert by_audio[se_nashi_suffix].existing_subtitle_path == se_ari.with_suffix(".srt")
    assert by_audio[se_nashi_prefix].action == "copy_existing_subtitle"
    assert by_audio[se_nashi_prefix].existing_subtitle_path == se_ari.with_suffix(".srt")


def test_same_work_fuzzy_near_duplicate_audio_name_is_treated_as_duplicate_content(tmp_path: Path):
    root = tmp_path / "root"
    original = touch(root / "work" / "01" / "01_プロローグ「純真魔法少女・アリス＆メスガキ魔法少女・シオン」.flac")
    bgm_off = touch(root / "work" / "02" / "【BGMなし】01_プロローグ「純真魔法少女・アリス＆メスガキ魔法少女・シオン」.wav")
    original.with_suffix(".srt").write_text("existing subtitle", encoding="utf-8")

    tasks = discover_audio_tasks(root)

    by_audio = {task.audio_path: task for task in tasks}
    assert by_audio[original].action == "skip"
    assert by_audio[bgm_off].action == "copy_existing_subtitle"
    assert by_audio[bgm_off].reason == "existing_duplicate_subtitle:.srt"
    assert by_audio[bgm_off].existing_subtitle_path == original.with_suffix(".srt")


def test_same_work_fuzzy_matching_does_not_merge_different_track_numbers(tmp_path: Path):
    root = tmp_path / "root"
    track1 = touch(root / "work" / "01" / "Track01_添い寝ボイス.flac")
    track2 = touch(root / "work" / "01" / "Track02_添い寝ボイス.flac")

    tasks = discover_audio_tasks(root)

    by_audio = {task.audio_path: task for task in tasks}
    assert by_audio[track1].action == "process"
    assert by_audio[track2].action == "process"


def test_same_work_fuzzy_matching_does_not_merge_similar_names_in_same_directory(tmp_path: Path):
    root = tmp_path / "root"
    first = touch(root / "work" / "02_特典" / "06_購入特典4_アイドル白崎心春のボイスアラーム1_朝.wav")
    second = touch(root / "work" / "02_特典" / "07_購入特典4_アイドル白崎心春のボイスアラーム2_昼.wav")
    third = touch(root / "work" / "02_特典" / "【BGMなし】06_購入特典4_アイドル白崎心春のボイスアラーム1_朝.wav")

    tasks = discover_audio_tasks(root)

    by_audio = {task.audio_path: task for task in tasks}
    assert by_audio[first].action == "process"
    assert by_audio[second].action == "process"
    assert by_audio[third].action == "process"


def test_intermediate_paths_are_same_directory_and_overwritable(tmp_path: Path):
    root = tmp_path / "root"
    audio = touch(root / "work♡" / "01" / "♡トラック01.wav")

    [task] = discover_audio_tasks(root)

    assert task.transcribe_srt_path == audio.with_name("♡トラック01-transcribe.srt")
    assert task.translate_srt_path == audio.with_name("♡トラック01-translate.srt")
    assert task.opencc_srt_path == audio.with_name("♡トラック01-opencc.srt")
