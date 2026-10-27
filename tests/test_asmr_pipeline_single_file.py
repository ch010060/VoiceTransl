from pathlib import Path

import pytest

from scripts.asmr_transcribe_lib.local_faster_whisper import FasterWhisperConfig
from scripts.asmr_transcribe_lib.pipeline import process_audio_task
from scripts.asmr_transcribe_lib.planner import discover_audio_tasks


def test_process_queued_audio_runs_transcriber_and_validates_transcribe_srt(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    [task] = discover_audio_tasks(root)
    calls = []

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        calls.append((audio_path, transcribe_srt_path, project_root, config))
        Path(transcribe_srt_path).write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n",
            encoding="utf-8",
        )
        return Path(transcribe_srt_path)

    result = process_audio_task(
        task,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        transcriber=fake_transcriber,
    )

    assert result.status == "transcribed"
    assert result.reason == "transcribe_complete"
    assert task.transcribe_srt_path.exists()
    assert not task.final_srt_path.exists()
    assert calls == [(task.audio_path, task.transcribe_srt_path, root, FasterWhisperConfig(model="model", language="ja"))]


def test_process_queued_audio_fails_when_transcribe_srt_is_invalid(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    [task] = discover_audio_tasks(root)

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        Path(transcribe_srt_path).write_text("bad subtitle", encoding="utf-8")
        return Path(transcribe_srt_path)

    with pytest.raises(ValueError, match="timestamp"):
        process_audio_task(
            task,
            project_root=root,
            faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
            transcriber=fake_transcriber,
        )

    assert not task.final_srt_path.exists()


def test_process_existing_srt_skip_does_not_overwrite(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.flac"
    final_srt = root / "work" / "01" / "track.srt"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    final_srt.write_text("existing", encoding="utf-8")
    [task] = discover_audio_tasks(root)

    def fail_if_called(**kwargs):
        raise AssertionError("transcriber should not be called")

    result = process_audio_task(
        task,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        transcriber=fail_if_called,
    )

    assert result.status == "skipped"
    assert result.reason == "existing_srt"
    assert final_srt.read_text(encoding="utf-8") == "existing"


def test_process_existing_alt_subtitle_copies_to_final_srt_without_conversion(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.mp3"
    vtt = root / "work" / "01" / "track.vtt"
    final_srt = root / "work" / "01" / "track.srt"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    vtt.write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n", encoding="utf-8")
    [task] = discover_audio_tasks(root)

    result = process_audio_task(
        task,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        transcriber=lambda **kwargs: (_ for _ in ()).throw(AssertionError("transcriber should not be called")),
    )

    assert result.status == "copied"
    assert result.reason == "existing_alt_subtitle:.vtt"
    assert final_srt.read_text(encoding="utf-8") == vtt.read_text(encoding="utf-8")


def test_process_existing_alt_subtitle_does_not_overwrite_final_srt_if_it_appears_late(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.mp3"
    vtt = root / "work" / "01" / "track.vtt"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    vtt.write_text("WEBVTT", encoding="utf-8")
    [task] = discover_audio_tasks(root)
    task.final_srt_path.write_text("late existing", encoding="utf-8")

    result = process_audio_task(
        task,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        transcriber=lambda **kwargs: (_ for _ in ()).throw(AssertionError("transcriber should not be called")),
    )

    assert result.status == "skipped"
    assert result.reason == "existing_srt"
    assert task.final_srt_path.read_text(encoding="utf-8") == "late existing"
