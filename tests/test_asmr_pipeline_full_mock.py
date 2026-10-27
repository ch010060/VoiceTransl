from pathlib import Path
import json

import pytest

from scripts.asmr_transcribe_lib.local_faster_whisper import FasterWhisperConfig
from scripts.asmr_transcribe_lib.pipeline import process_audio_task
from scripts.asmr_transcribe_lib.planner import discover_audio_tasks


def test_process_audio_task_full_mock_pipeline_writes_intermediates_and_final_traditional_srt(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    [task] = discover_audio_tasks(root)
    calls = []

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        calls.append(("transcribe", audio_path, transcribe_srt_path))
        Path(transcribe_srt_path).write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n雨の音。\n",
            encoding="utf-8",
        )
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        calls.append(("translate", source_srt_path, target_srt_path))
        Path(target_srt_path).write_text(
            "1\n00:00:01,000 --> 00:00:02,000\n你好。\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n能听见雨声。\n",
            encoding="utf-8",
        )
        return Path(target_srt_path)

    result = process_audio_task(
        task,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        transcriber=fake_transcriber,
        translator=fake_translator,
    )

    assert result.status == "success"
    assert result.reason == "pipeline_complete"
    assert task.transcribe_srt_path.exists()
    assert task.translate_srt_path.exists()
    assert task.opencc_srt_path.exists()
    assert task.final_srt_path.exists()
    assert "能聽見雨聲" in task.final_srt_path.read_text(encoding="utf-8")
    assert task.final_srt_path.read_text(encoding="utf-8") == task.opencc_srt_path.read_text(encoding="utf-8")
    assert [call[0] for call in calls] == ["transcribe", "translate"]


def test_full_pipeline_removes_stale_intermediates_before_rerun(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    [task] = discover_audio_tasks(root)
    task.transcribe_srt_path.write_text("stale transcribe", encoding="utf-8")
    task.translate_srt_path.write_text("stale translate", encoding="utf-8")
    task.opencc_srt_path.write_text("stale opencc", encoding="utf-8")

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        assert not task.transcribe_srt_path.exists()
        assert not task.translate_srt_path.exists()
        assert not task.opencc_srt_path.exists()
        Path(transcribe_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n汉字。\n", encoding="utf-8")
        return Path(target_srt_path)

    process_audio_task(
        task,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        transcriber=fake_transcriber,
        translator=fake_translator,
    )

    assert "stale" not in task.final_srt_path.read_text(encoding="utf-8")
    assert "漢字" in task.final_srt_path.read_text(encoding="utf-8")


def test_full_pipeline_rerun_ignores_stale_final_tmp_file(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    [task] = discover_audio_tasks(root)
    stale_tmp = task.final_srt_path.with_name(f"{task.final_srt_path.name}.tmp")
    stale_tmp.write_text("interrupted stale final tmp", encoding="utf-8")

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        Path(transcribe_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n汉字。\n", encoding="utf-8")
        return Path(target_srt_path)

    process_audio_task(
        task,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        transcriber=fake_transcriber,
        translator=fake_translator,
    )

    assert not stale_tmp.exists()
    assert task.final_srt_path.exists()
    assert "漢字" in task.final_srt_path.read_text(encoding="utf-8")
    assert "interrupted stale" not in task.final_srt_path.read_text(encoding="utf-8")


def test_full_pipeline_applies_transcript_corrections_before_translation(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "asmr_transcript_corrections.json").write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "wrong": "作成",
                        "correct": "搾精",
                        "include_any": ["サキュバス"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    [task] = discover_audio_tasks(root)
    seen_source_text = []

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        Path(transcribe_srt_path).write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nクロエも、立派なサキュバスなのだから、ちゃんと作成もできるようにならねばな。\n",
            encoding="utf-8",
        )
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        seen_source_text.append(Path(source_srt_path).read_text(encoding="utf-8"))
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n你也要學會榨精。\n", encoding="utf-8")
        return Path(target_srt_path)

    process_audio_task(
        task,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        transcriber=fake_transcriber,
        translator=fake_translator,
    )

    assert "搾精" in seen_source_text[0]
    assert "作成" not in seen_source_text[0]
    assert "搾精" in task.transcribe_srt_path.read_text(encoding="utf-8")


def test_full_pipeline_applies_traditional_chinese_corrections_after_opencc(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "asmr_traditional_chinese_corrections.json").write_text(
        json.dumps(
            {
                "rules": [
                    {"wrong": "想象", "correct": "想像"},
                    {"wrong": "好惡心", "correct": "好噁心"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    [task] = discover_audio_tasks(root)

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        Path(transcribe_srt_path).write_text(
            "1\n00:00:01,000 --> 00:00:02,000\n何を想像しているんですか。きも、気持ち悪い。\n",
            encoding="utf-8",
        )
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        Path(target_srt_path).write_text(
            "1\n00:00:01,000 --> 00:00:02,000\n你在想象什么？好恶心。\n",
            encoding="utf-8",
        )
        return Path(target_srt_path)

    process_audio_task(
        task,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        transcriber=fake_transcriber,
        translator=fake_translator,
    )

    assert "你在想像什麼？好噁心。" in task.opencc_srt_path.read_text(encoding="utf-8")
    assert task.final_srt_path.read_text(encoding="utf-8") == task.opencc_srt_path.read_text(encoding="utf-8")


def test_full_pipeline_translation_failure_does_not_create_final_srt(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    [task] = discover_audio_tasks(root)

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        Path(transcribe_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        return Path(transcribe_srt_path)

    def failing_translator(source_srt_path, target_srt_path):
        raise RuntimeError("translation failed")

    with pytest.raises(RuntimeError, match="translation failed"):
        process_audio_task(
            task,
            project_root=root,
            faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
            transcriber=fake_transcriber,
            translator=failing_translator,
        )

    assert task.transcribe_srt_path.exists()
    assert not task.translate_srt_path.exists()
    assert not task.opencc_srt_path.exists()
    assert not task.final_srt_path.exists()


def test_full_pipeline_never_overwrites_existing_final_srt(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.wav"
    final_srt = root / "work" / "01" / "track.srt"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    final_srt.write_text("existing final", encoding="utf-8")
    [task] = discover_audio_tasks(root)

    result = process_audio_task(
        task,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        transcriber=lambda **kwargs: (_ for _ in ()).throw(AssertionError("transcriber should not run")),
        translator=lambda source_srt_path, target_srt_path: (_ for _ in ()).throw(AssertionError("translator should not run")),
    )

    assert result.status == "skipped"
    assert final_srt.read_text(encoding="utf-8") == "existing final"


def test_resume_guard_skips_retranscription_when_valid_transcribe_srt_exists(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    [task] = discover_audio_tasks(root)
    task.transcribe_srt_path.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n",
        encoding="utf-8",
    )
    task.translate_srt_path.write_text("stale translate", encoding="utf-8")
    calls = []

    def fail_if_transcribed(**kwargs):
        raise AssertionError("transcriber must not be called when valid transcribe.srt exists")

    def fake_translator(source_srt_path, target_srt_path):
        calls.append("translate")
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n你好。\n", encoding="utf-8")
        return Path(target_srt_path)

    result = process_audio_task(
        task,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        transcriber=fail_if_transcribed,
        translator=fake_translator,
    )

    assert result.status == "success"
    assert calls == ["translate"]
    assert not task.translate_srt_path.exists() or task.translate_srt_path.read_text(encoding="utf-8") != "stale translate"
    assert task.final_srt_path.exists()


def test_resume_guard_retranscribes_when_transcribe_srt_is_unparseable(tmp_path: Path):
    root = tmp_path / "root"
    audio = root / "work" / "01" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    [task] = discover_audio_tasks(root)
    task.transcribe_srt_path.write_text("not valid srt content", encoding="utf-8")
    calls = []

    def fake_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
        calls.append("transcribe")
        assert not Path(transcribe_srt_path).exists(), "stale invalid transcribe.srt must be removed before transcriber call"
        Path(transcribe_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
        return Path(transcribe_srt_path)

    def fake_translator(source_srt_path, target_srt_path):
        Path(target_srt_path).write_text("1\n00:00:01,000 --> 00:00:02,000\n你好。\n", encoding="utf-8")
        return Path(target_srt_path)

    result = process_audio_task(
        task,
        project_root=root,
        faster_whisper_config=FasterWhisperConfig(model="model", language="ja"),
        transcriber=fake_transcriber,
        translator=fake_translator,
    )

    assert result.status == "success"
    assert calls == ["transcribe"]
