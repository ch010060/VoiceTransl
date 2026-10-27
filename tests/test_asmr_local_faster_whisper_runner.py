from pathlib import Path
import subprocess

import pytest

from scripts.asmr_transcribe_lib.local_faster_whisper import (
    FasterWhisperConfig,
    run_faster_whisper_transcription,
)


def test_run_faster_whisper_moves_mock_output_to_transcribe_srt(tmp_path: Path):
    project_root = tmp_path
    exe = project_root / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    audio = project_root / "work" / "track name.wav"
    final_transcribe = audio.with_name("track name-transcribe.srt")
    exe.parent.mkdir(parents=True)
    audio.parent.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")
    calls = []

    def fake_run(argv, cwd, check, capture_output, text, encoding, errors):
        calls.append({"argv": argv, "cwd": cwd, "check": check, "capture_output": capture_output, "text": text})
        assert encoding == "utf-8"
        assert errors == "replace"
        output_dir = Path(argv[argv.index("--output_dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "track name.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    result = run_faster_whisper_transcription(
        audio_path=audio,
        transcribe_srt_path=final_transcribe,
        project_root=project_root,
        config=FasterWhisperConfig(model="faster-whisper-large-v3-turbo-ct2", language="ja"),
        subprocess_run=fake_run,
    )

    assert result == final_transcribe
    assert final_transcribe.read_text(encoding="utf-8").startswith("1\n00:00:01,000")
    assert len(calls) == 1
    argv = calls[0]["argv"]
    assert argv[0] == str(exe)
    assert str(audio) in argv
    assert "--model" in argv
    assert "large-v3-turbo-ct2" in argv
    assert "--language" in argv
    assert "ja" in argv
    assert "--output_format" in argv
    assert "srt" in argv
    assert calls[0]["cwd"] == project_root
    assert calls[0]["check"] is False


def test_run_faster_whisper_replaces_stale_intermediate_before_running(tmp_path: Path):
    project_root = tmp_path
    exe = project_root / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    audio = project_root / "work" / "track.wav"
    final_transcribe = audio.with_name("track-transcribe.srt")
    exe.parent.mkdir(parents=True)
    audio.parent.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")
    final_transcribe.write_text("stale", encoding="utf-8")

    def fake_run(argv, cwd, check, capture_output, text, encoding, errors):
        assert not final_transcribe.exists()
        output_dir = Path(argv[argv.index("--output_dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "track.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\n新しい。\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    run_faster_whisper_transcription(
        audio_path=audio,
        transcribe_srt_path=final_transcribe,
        project_root=project_root,
        config=FasterWhisperConfig(model="model", language="ja"),
        subprocess_run=fake_run,
    )

    assert "新しい" in final_transcribe.read_text(encoding="utf-8")


def test_run_faster_whisper_uses_ascii_safe_temp_output_dir_for_unicode_audio_path(tmp_path: Path):
    project_root = tmp_path
    exe = project_root / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    audio = project_root / "作品" / "雨・猫.wav"
    final_transcribe = audio.with_name("雨・猫-transcribe.srt")
    exe.parent.mkdir(parents=True)
    audio.parent.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")

    def fake_run(argv, cwd, check, capture_output, text, encoding, errors):
        output_dir = Path(argv[argv.index("--output_dir") + 1])
        assert output_dir.parent != audio.parent.resolve()
        assert output_dir.parent == (project_root / ".asmr_tmp").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "雨・猫.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\n新しい。\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    run_faster_whisper_transcription(
        audio_path=audio,
        transcribe_srt_path=final_transcribe,
        project_root=project_root,
        config=FasterWhisperConfig(model="model", language="ja"),
        subprocess_run=fake_run,
    )

    assert final_transcribe.exists()


def test_run_faster_whisper_accepts_nonzero_exit_when_srt_was_written(tmp_path: Path):
    project_root = tmp_path
    exe = project_root / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    audio = project_root / "work" / "track.wav"
    final_transcribe = audio.with_name("track-transcribe.srt")
    exe.parent.mkdir(parents=True)
    audio.parent.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")

    def fake_run(argv, cwd, check, capture_output, text, encoding, errors):
        assert check is False
        output_dir = Path(argv[argv.index("--output_dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "track.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\n新しい。\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 3221226505, stdout="wrote file", stderr="late native crash")

    result = run_faster_whisper_transcription(
        audio_path=audio,
        transcribe_srt_path=final_transcribe,
        project_root=project_root,
        config=FasterWhisperConfig(model="model", language="ja"),
        subprocess_run=fake_run,
    )

    assert result == final_transcribe
    assert final_transcribe.exists()


def test_run_faster_whisper_fails_when_expected_srt_missing(tmp_path: Path):
    project_root = tmp_path
    exe = project_root / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    audio = project_root / "work" / "track.wav"
    final_transcribe = audio.with_name("track-transcribe.srt")
    exe.parent.mkdir(parents=True)
    audio.parent.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")

    def fake_run(argv, cwd, check, capture_output, text, encoding, errors):
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    with pytest.raises(FileNotFoundError, match="transcription SRT output"):
        run_faster_whisper_transcription(
            audio_path=audio,
            transcribe_srt_path=final_transcribe,
            project_root=project_root,
            config=FasterWhisperConfig(model="model", language="ja"),
            subprocess_run=fake_run,
        )
