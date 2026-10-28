from pathlib import Path

import pytest

from scripts.asmr_transcribe_lib.local_faster_whisper import (
    FasterWhisperConfig,
    build_faster_whisper_argv,
    find_faster_whisper_exe,
    preflight_faster_whisper,
    preflight_faster_whisper_model,
    resolve_faster_whisper_model_arg,
)


def test_finds_unrenamed_faster_whisper_xxl_exe(tmp_path: Path):
    exe = tmp_path / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")

    assert find_faster_whisper_exe(tmp_path) == exe


def test_prefers_gui_whisper_faster_exe_when_present(tmp_path: Path):
    gui_exe = tmp_path / "whisper-faster" / "whisper-faster.exe"
    xxl_exe = tmp_path / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    gui_exe.parent.mkdir(parents=True)
    xxl_exe.parent.mkdir(parents=True)
    gui_exe.write_text("gui", encoding="utf-8")
    xxl_exe.write_text("xxl", encoding="utf-8")

    assert find_faster_whisper_exe(tmp_path) == gui_exe


def test_preflight_reports_missing_exe(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="faster-whisper executable"):
        preflight_faster_whisper(tmp_path)


def test_builds_faster_whisper_xxl_argv_without_shell(tmp_path: Path):
    exe = tmp_path / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    audio = tmp_path / "work" / "track name.wav"
    output_dir = tmp_path / "work"
    exe.parent.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")
    model_dir = tmp_path / "whisper-faster"
    config = FasterWhisperConfig(model="faster-whisper-large-v3-turbo-ct2", language="ja", model_dir=model_dir)

    argv = build_faster_whisper_argv(exe, audio, output_dir, config)

    assert argv[0] == str(exe)
    assert str(audio) in argv
    assert "--model" in argv
    assert "large-v3-turbo-ct2" in argv
    assert "--model_dir" in argv
    assert str(model_dir) in argv
    assert "--language" in argv
    assert "ja" in argv
    assert "--output_format" in argv
    assert "srt" in argv
    assert "--output_dir" in argv
    assert str(output_dir) in argv


def test_resolve_gui_faster_whisper_model_name_for_xxl_binary():
    assert resolve_faster_whisper_model_arg("faster-whisper-large-v3-turbo-ct2") == "large-v3-turbo-ct2"
    assert resolve_faster_whisper_model_arg("large-v3-turbo") == "large-v3-turbo"


def test_preflight_reports_missing_local_ct2_model_folder(tmp_path: Path):
    exe = tmp_path / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    (tmp_path / "whisper-faster").mkdir()

    with pytest.raises(FileNotFoundError, match="faster-whisper-large-v3-turbo-ct2"):
        preflight_faster_whisper_model(tmp_path, "faster-whisper-large-v3-turbo-ct2")


def test_preflight_accepts_local_ct2_model_folder(tmp_path: Path):
    exe = tmp_path / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    model_folder = tmp_path / "whisper-faster" / "faster-whisper-large-v3-turbo-ct2"
    exe.parent.mkdir(parents=True)
    model_folder.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")

    assert preflight_faster_whisper_model(tmp_path, "faster-whisper-large-v3-turbo-ct2") == model_folder


def test_preflight_prefers_local_folder_for_known_xxl_model(tmp_path: Path):
    exe = tmp_path / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    model_folder = tmp_path / "whisper-faster" / "faster-whisper-large-v3"
    exe.parent.mkdir(parents=True)
    model_folder.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")

    assert preflight_faster_whisper_model(tmp_path, "faster-whisper-large-v3") == model_folder


def test_builds_argv_with_explicit_vad_and_word_timestamps_by_default(tmp_path: Path):
    exe = tmp_path / "whisper-faster" / "whisper-faster.exe"
    audio = tmp_path / "track.wav"
    output_dir = tmp_path / "out"
    exe.parent.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")
    config = FasterWhisperConfig(model="large-v3", language="ja")

    argv = build_faster_whisper_argv(exe, audio, output_dir, config)

    assert "--vad_filter" in argv
    assert argv[argv.index("--vad_filter") + 1] == "True"
    assert "--word_timestamps" in argv
    assert argv[argv.index("--word_timestamps") + 1] == "True"
    assert "--batched" not in argv


def test_builds_argv_with_batched_preset_when_enabled(tmp_path: Path):
    exe = tmp_path / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    audio = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    exe.parent.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")
    config = FasterWhisperConfig(model="large-v3", language="ja", batched=True, batch_size=16, unmerged=True)

    argv = build_faster_whisper_argv(exe, audio, output_dir, config)

    assert "--batched" in argv
    assert "--batch_size" in argv
    assert argv[argv.index("--batch_size") + 1] == "16"
    assert "--unmerged" in argv


def test_builds_argv_with_initial_prompt(tmp_path: Path):
    exe = tmp_path / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    audio = tmp_path / "track.wav"
    output_dir = tmp_path / "out"
    exe.parent.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")
    config = FasterWhisperConfig(
        model="large-v3",
        language="ja",
        initial_prompt="サキュバスが搾精し射精させます。",
        reprompt=True,
    )

    argv = build_faster_whisper_argv(exe, audio, output_dir, config)

    assert "--initial_prompt" in argv
    assert argv[argv.index("--initial_prompt") + 1] == "サキュバスが搾精し射精させます。"
    assert "--reprompt" in argv
    assert argv[argv.index("--reprompt") + 1] == "True"
    assert "--ignore_dupe_prompt" not in argv


def test_builds_argv_without_prompt_when_initial_prompt_is_none(tmp_path: Path):
    exe = tmp_path / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    audio = tmp_path / "track.wav"
    output_dir = tmp_path / "out"
    exe.parent.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")
    config = FasterWhisperConfig(model="large-v3", language="ja")

    argv = build_faster_whisper_argv(exe, audio, output_dir, config)

    assert "--initial_prompt" not in argv
    assert "--reprompt" not in argv


def test_builds_argv_with_batched_without_unmerged(tmp_path: Path):
    exe = tmp_path / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    audio = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    exe.parent.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")
    config = FasterWhisperConfig(model="large-v3", language="ja", batched=True, unmerged=False)

    argv = build_faster_whisper_argv(exe, audio, output_dir, config)

    assert "--batched" in argv
    assert "--unmerged" not in argv


def test_builds_argv_with_sentence_and_chunk_length(tmp_path: Path):
    exe = tmp_path / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    audio = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    exe.parent.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")
    config = FasterWhisperConfig(model="large-v3", language="ja", batched=True, batch_size=8, sentence=True, chunk_length=20)

    argv = build_faster_whisper_argv(exe, audio, output_dir, config)

    assert argv[argv.index("--batch_size") + 1] == "8"
    assert argv[argv.index("--chunk_length") + 1] == "20"
    assert "--sentence" in argv
    assert "--unmerged" not in argv


def test_sequential_argv_has_no_batched_only_flags(tmp_path: Path):
    exe = tmp_path / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"
    audio = tmp_path / "track.flac"
    output_dir = tmp_path / "out"
    exe.parent.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    exe.write_text("exe", encoding="utf-8")
    audio.write_text("audio", encoding="utf-8")
    # chunk_length only applies to batched mode
    config = FasterWhisperConfig(model="large-v3", language="ja", chunk_length=20)

    argv = build_faster_whisper_argv(exe, audio, output_dir, config)

    assert "--chunk_length" not in argv
    assert "--sentence" not in argv
    assert "--batched" not in argv
