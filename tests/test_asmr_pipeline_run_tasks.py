from pathlib import Path
import threading
import time

from scripts.asmr_transcribe_lib.local_faster_whisper import FasterWhisperConfig
from scripts.asmr_transcribe_lib.pipeline import run_tasks
from scripts.asmr_transcribe_lib.planner import discover_audio_tasks


def test_run_tasks_respects_limit_and_processes_in_plan_order(tmp_path: Path):
    root = tmp_path / "root"
    first = root / "work_a" / "01.wav"
    second = root / "work_a" / "02.wav"
    first.parent.mkdir(parents=True)
    first.write_text("audio", encoding="utf-8")
    second.write_text("audio", encoding="utf-8")
    tasks = discover_audio_tasks(root)
    processed = []

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        processed.append(Path(audio_path).name)
        Path(transcribe_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n汉字。\n", encoding="utf-8")
        return Path(target_srt_path)

    results = run_tasks(
        tasks,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        translator=fake_translator,
        transcriber=fake_transcriber,
        limit=1,
    )

    assert processed == ["01.wav"]
    assert [result.status for result in results] == ["success"]
    assert first.with_suffix(".srt").exists()
    assert not second.with_suffix(".srt").exists()


def test_run_tasks_includes_skipped_tasks_without_consuming_limit(tmp_path: Path):
    root = tmp_path / "root"
    skipped_audio = root / "work" / "done.wav"
    queued_audio = root / "work" / "todo.wav"
    skipped_audio.parent.mkdir(parents=True)
    skipped_audio.write_text("audio", encoding="utf-8")
    skipped_audio.with_suffix(".srt").write_text("existing", encoding="utf-8")
    queued_audio.write_text("audio", encoding="utf-8")
    tasks = discover_audio_tasks(root)

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        Path(transcribe_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n汉字。\n", encoding="utf-8")
        return Path(target_srt_path)

    results = run_tasks(
        tasks,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        translator=fake_translator,
        transcriber=fake_transcriber,
        limit=1,
    )

    assert [result.status for result in results] == ["skipped", "success"]
    assert queued_audio.with_suffix(".srt").exists()


def test_run_tasks_defaults_to_single_concurrent_transcription_even_with_parallel_workers(tmp_path: Path):
    root = tmp_path / "root"
    work = root / "work"
    work.mkdir(parents=True)
    for index in range(3):
        (work / f"{index}.wav").write_text("audio", encoding="utf-8")
    tasks = discover_audio_tasks(root)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        Path(transcribe_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        with lock:
            active -= 1
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n汉字。\n", encoding="utf-8")
        return Path(target_srt_path)

    results = run_tasks(
        tasks,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        translator=fake_translator,
        transcriber=fake_transcriber,
        parallel=2,
    )

    assert max_active == 1
    assert [result.status for result in results] == ["success", "success", "success"]
    assert [result.audio_path.name for result in results] == ["0.wav", "1.wav", "2.wav"]


def test_run_tasks_reports_running_and_done_progress_events(tmp_path: Path):
    root = tmp_path / "root"
    work = root / "work"
    work.mkdir(parents=True)
    for index in range(2):
        (work / f"{index}.wav").write_text("audio", encoding="utf-8")
    tasks = discover_audio_tasks(root)
    events = []

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        Path(transcribe_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n汉字。\n", encoding="utf-8")
        return Path(target_srt_path)

    run_tasks(
        tasks,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        translator=fake_translator,
        transcriber=fake_transcriber,
        parallel=2,
        progress_callback=lambda index, task, status, result: events.append((index, task.audio_path.name, status, result.status if result else None)),
    )

    assert (0, "0.wav", "running", None) in events
    assert (1, "1.wav", "running", None) in events
    assert (0, "0.wav", "success", "success") in events
    assert (1, "1.wav", "success", "success") in events


def test_run_tasks_reports_running_only_after_transcribe_slot_is_acquired(tmp_path: Path):
    root = tmp_path / "root"
    work = root / "work"
    work.mkdir(parents=True)
    for index in range(3):
        (work / f"{index}.wav").write_text("audio", encoding="utf-8")
    tasks = discover_audio_tasks(root)
    events = []
    events_lock = threading.Lock()
    first_transcriber_entered = threading.Event()
    release_transcriber = threading.Event()

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        first_transcriber_entered.set()
        assert release_transcriber.wait(timeout=5)
        Path(transcribe_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n汉字。\n", encoding="utf-8")
        return Path(target_srt_path)

    def progress(index, task, status, result):
        del task, result
        with events_lock:
            events.append((index, status))

    runner = threading.Thread(
        target=run_tasks,
        kwargs={
            "tasks": tasks,
            "project_root": root,
            "faster_whisper_config": FasterWhisperConfig(model="model", language="ja"),
            "translator": fake_translator,
            "transcriber": fake_transcriber,
            "parallel": 3,
            "transcribe_parallel": 1,
            "progress_callback": progress,
        },
    )
    runner.start()
    assert first_transcriber_entered.wait(timeout=5)
    time.sleep(0.05)

    with events_lock:
        running_events_before_release = [event for event in events if event[1] == "running"]

    release_transcriber.set()
    runner.join(timeout=5)

    assert not runner.is_alive()
    assert len(running_events_before_release) == 1


def test_run_tasks_appends_progress_events_as_processing_continues(tmp_path: Path, capsys):
    from scripts.asmr_transcribe_lib.cli import print_progress_snapshot

    root = tmp_path / "root"
    work = root / "work"
    work.mkdir(parents=True)
    for index in range(3):
        (work / f"{index}.wav").write_text("audio", encoding="utf-8")
    tasks = discover_audio_tasks(root)
    statuses = [task.status for task in tasks]

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        Path(transcribe_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n汉字。\n", encoding="utf-8")
        return Path(target_srt_path)

    def progress(index, task, status, result):
        del task, result
        statuses[index] = status
        display_tasks = [
            task if task.status == statuses[item_index] else task.__class__(**{**task.__dict__, "status": statuses[item_index]})
            for item_index, task in enumerate(tasks)
        ]
        print_progress_snapshot(display_tasks, parallel=2, elapsed_seconds=0)

    run_tasks(
        tasks,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        translator=fake_translator,
        transcriber=fake_transcriber,
        parallel=2,
        progress_callback=progress,
    )

    output = capsys.readouterr().out
    assert output.count("#   Status") >= 2
    assert "\033[2J" not in output


def test_run_tasks_can_raise_transcription_concurrency_when_explicitly_requested(tmp_path: Path):
    root = tmp_path / "root"
    work = root / "work"
    work.mkdir(parents=True)
    for index in range(3):
        (work / f"{index}.wav").write_text("audio", encoding="utf-8")
    tasks = discover_audio_tasks(root)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        Path(transcribe_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        with lock:
            active -= 1
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n汉字。\n", encoding="utf-8")
        return Path(target_srt_path)

    results = run_tasks(
        tasks,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        translator=fake_translator,
        transcriber=fake_transcriber,
        parallel=2,
        transcribe_parallel=2,
    )

    assert max_active == 2
    assert [result.status for result in results] == ["success", "success", "success"]
    assert [result.audio_path.name for result in results] == ["0.wav", "1.wav", "2.wav"]


def test_run_tasks_processes_duplicate_audio_once_then_copies_generated_srt(tmp_path: Path):
    root = tmp_path / "root"
    mp3 = root / "work" / "mp3" / "Track01.mp3"
    flac = root / "work" / "flac" / "Track01.flac"
    mp3.parent.mkdir(parents=True)
    flac.parent.mkdir(parents=True)
    mp3.write_text("audio", encoding="utf-8")
    flac.write_text("audio", encoding="utf-8")
    tasks = discover_audio_tasks(root)
    processed = []

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        processed.append(Path(audio_path))
        Path(transcribe_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n汉字。\n", encoding="utf-8")
        return Path(target_srt_path)

    results = run_tasks(
        tasks,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        translator=fake_translator,
        transcriber=fake_transcriber,
        parallel=2,
    )

    assert processed == [flac]
    assert flac.with_suffix(".srt").exists()
    assert mp3.with_suffix(".srt").read_text(encoding="utf-8") == flac.with_suffix(".srt").read_text(encoding="utf-8")
    assert sorted(result.status for result in results) == ["copied", "success"]


def test_rerun_after_interruption_copies_duplicate_from_existing_representative_srt(tmp_path: Path):
    root = tmp_path / "root"
    mp3 = root / "work" / "mp3" / "Track01.mp3"
    flac = root / "work" / "flac" / "Track01.flac"
    mp3.parent.mkdir(parents=True)
    flac.parent.mkdir(parents=True)
    mp3.write_text("audio", encoding="utf-8")
    flac.write_text("audio", encoding="utf-8")
    flac.with_suffix(".srt").write_text("representative final subtitle", encoding="utf-8")

    tasks = discover_audio_tasks(root)

    def fail_if_transcribed(**kwargs):
        raise AssertionError("rerun should copy from existing representative subtitle, not transcribe again")

    results = run_tasks(
        tasks,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        translator=lambda source_srt_path, target_srt_path: (_ for _ in ()).throw(AssertionError("translator should not run")),
        transcriber=fail_if_transcribed,
        parallel=2,
    )

    assert mp3.with_suffix(".srt").read_text(encoding="utf-8") == "representative final subtitle"
    assert sorted(result.status for result in results) == ["copied", "skipped"]
