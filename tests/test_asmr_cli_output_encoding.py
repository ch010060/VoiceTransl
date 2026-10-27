import sys
from dataclasses import replace
from pathlib import Path

from scripts.asmr_transcribe_lib import cli
from scripts.asmr_transcribe_lib.cli import print_progress_snapshot
from scripts.asmr_transcribe_lib.pipeline import TaskResult
from scripts.asmr_transcribe_lib.models import AudioTask


class Cp950StrictStdout:
    encoding = "cp950"

    def __init__(self):
        self.parts = []

    def write(self, text: str):
        text.encode(self.encoding)
        self.parts.append(text)

    def flush(self):
        pass


def test_progress_output_replaces_characters_not_supported_by_stdout_encoding(monkeypatch, tmp_path: Path):
    audio = tmp_path / "work" / "♡トラック.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    stdout = Cp950StrictStdout()
    monkeypatch.setattr(sys, "stdout", stdout)
    task = AudioTask(
        work_name="作品♡トラック",
        work_dir=audio.parent.parent,
        audio_path=audio,
        root_relative_audio_path=Path("作品♡トラック/♡トラック.wav"),
        relative_audio_path=Path("作品♡トラック/♡トラック.wav"),
        final_srt_path=audio.with_suffix(".srt"),
        transcribe_srt_path=audio.with_name("♡トラック-transcribe.srt"),
        translate_srt_path=audio.with_name("♡トラック-translate.srt"),
        opencc_srt_path=audio.with_name("♡トラック-opencc.srt"),
    )

    print_progress_snapshot([task], parallel=1, elapsed_seconds=0)

    output = "".join(stdout.parts)
    assert "#   Status" in output
    assert "Stage" in output
    assert "Time" in output
    assert "queued" in output
    assert "\\u2661" in output


def test_progress_snapshot_displays_only_parallel_sized_window(capsys, tmp_path: Path):
    tasks = []
    for index in range(5):
        audio = tmp_path / "work" / f"{index}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_text("audio", encoding="utf-8")
        tasks.append(
            AudioTask(
                work_name="work",
                work_dir=audio.parent,
                audio_path=audio,
                root_relative_audio_path=Path(f"work/{index}.wav"),
                relative_audio_path=Path(f"work/{index}.wav"),
                final_srt_path=audio.with_suffix(".srt"),
                transcribe_srt_path=audio.with_name(f"{index}-transcribe.srt"),
                translate_srt_path=audio.with_name(f"{index}-translate.srt"),
                opencc_srt_path=audio.with_name(f"{index}-opencc.srt"),
            )
        )

    print_progress_snapshot(tasks, parallel=2, elapsed_seconds=0)

    output = capsys.readouterr().out
    assert "1   queued" in output
    assert "2   queued" in output
    assert "3   queued" not in output
    assert "more audio tasks hidden" not in output


def test_progress_snapshot_slides_forward_after_completed_task(capsys, tmp_path: Path):
    tasks = []
    for index in range(5):
        audio = tmp_path / "work" / f"{index}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_text("audio", encoding="utf-8")
        tasks.append(
            AudioTask(
                work_name="work",
                work_dir=audio.parent,
                audio_path=audio,
                root_relative_audio_path=Path(f"work/{index}.wav"),
                relative_audio_path=Path(f"work/{index}.wav"),
                final_srt_path=audio.with_suffix(".srt"),
                transcribe_srt_path=audio.with_name(f"{index}-transcribe.srt"),
                translate_srt_path=audio.with_name(f"{index}-translate.srt"),
                opencc_srt_path=audio.with_name(f"{index}-opencc.srt"),
            )
        )
    tasks[0] = replace(tasks[0], status="success")

    print_progress_snapshot(tasks, parallel=2, elapsed_seconds=0)

    output = capsys.readouterr().out
    assert "2   queued" in output
    assert "3   queued" in output
    assert "1   success" not in output
    assert "4   queued" not in output
    assert "more audio tasks hidden" not in output


def test_progress_snapshot_numbers_visible_tasks_without_gaps_when_skipped_tasks_exist(capsys, tmp_path: Path):
    tasks = []
    for index in range(4):
        audio = tmp_path / "work" / f"{index}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_text("audio", encoding="utf-8")
        tasks.append(
            AudioTask(
                work_name="work",
                work_dir=audio.parent,
                audio_path=audio,
                root_relative_audio_path=Path(f"work/{index}.wav"),
                relative_audio_path=Path(f"work/{index}.wav"),
                final_srt_path=audio.with_suffix(".srt"),
                transcribe_srt_path=audio.with_name(f"{index}-transcribe.srt"),
                translate_srt_path=audio.with_name(f"{index}-translate.srt"),
                opencc_srt_path=audio.with_name(f"{index}-opencc.srt"),
            )
        )
    tasks[1] = replace(tasks[1], status="skipped", action="skip")

    print_progress_snapshot(tasks, parallel=3, elapsed_seconds=0)

    output = capsys.readouterr().out
    assert "1   queued" in output
    assert "2   queued" in output
    assert "3   queued" in output
    assert "4   " not in output


def test_execute_cli_does_not_register_periodic_progress_tick_callback(monkeypatch, capsys, tmp_path: Path):
    audio = tmp_path / "root" / "work" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    task = AudioTask(
        work_name="work",
        work_dir=audio.parent,
        audio_path=audio,
        root_relative_audio_path=Path("work/track.wav"),
        relative_audio_path=Path("work/track.wav"),
        final_srt_path=audio.with_suffix(".srt"),
        transcribe_srt_path=audio.with_name("track-transcribe.srt"),
        translate_srt_path=audio.with_name("track-translate.srt"),
        opencc_srt_path=audio.with_name("track-opencc.srt"),
    )
    captured_kwargs = {}

    monkeypatch.setattr(cli, "discover_audio_tasks", lambda root: [task])
    monkeypatch.setattr(cli, "preflight_model", lambda base_url, model: None)
    monkeypatch.setattr(cli, "preflight_faster_whisper", lambda project_root: tmp_path / "fw.exe")
    monkeypatch.setattr(cli, "preflight_faster_whisper_model", lambda project_root, model: tmp_path / "model")
    monkeypatch.setattr(cli, "find_faster_whisper_exe", lambda project_root: tmp_path / "fw.exe")
    monkeypatch.setattr(cli, "find_faster_whisper_model_dir", lambda project_root: tmp_path)

    def fake_run_tasks(*args, **kwargs):
        del args
        captured_kwargs.update(kwargs)
        return [TaskResult(status="success", reason="ok", audio_path=task.audio_path, output_path=task.final_srt_path)]

    monkeypatch.setattr(cli, "run_tasks", fake_run_tasks)

    exit_code = cli.main(["--path", str(tmp_path / "root"), "--execute"])

    assert exit_code == 0
    assert "progress_tick_callback" not in captured_kwargs
    assert capsys.readouterr().out.count("#   Status") == 1

