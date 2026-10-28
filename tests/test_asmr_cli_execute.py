from pathlib import Path

from scripts.asmr_transcribe_lib import cli
from scripts.asmr_transcribe_lib.cli import DEFAULT_ASMR_INITIAL_PROMPT, _effective_initial_prompt
from scripts.asmr_transcribe_lib.pipeline import TaskResult


def test_cli_execute_preflights_and_runs_limited_tasks(monkeypatch, tmp_path: Path, capsys):
    root = tmp_path / "root"
    audio = root / "work" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    calls = []

    def fake_preflight_fw(project_root):
        calls.append(("fw", Path(project_root).name))
        return Path(project_root) / "Faster-Whisper-XXL" / "faster-whisper-xxl.exe"

    def fake_preflight_model(base_url, model):
        calls.append(("model", base_url, model))

    def fake_preflight_fw_model(project_root, model):
        calls.append(("fw_model", model))
        return Path(project_root) / "whisper-faster" / model

    def fake_run_tasks(tasks, **kwargs):
        calls.append(
            (
                "run",
                len(tasks),
                kwargs["limit"],
                kwargs["parallel"],
                kwargs["transcribe_parallel"],
                kwargs["faster_whisper_config"].language,
            )
        )
        return [TaskResult(status="success", reason="pipeline_complete", audio_path=tasks[0].audio_path, output_path=tasks[0].final_srt_path)]

    monkeypatch.setattr(cli, "preflight_faster_whisper", fake_preflight_fw)
    monkeypatch.setattr(cli, "preflight_faster_whisper_model", fake_preflight_fw_model)
    monkeypatch.setattr(cli, "preflight_model", fake_preflight_model)
    monkeypatch.setattr(cli, "run_tasks", fake_run_tasks)

    exit_code = cli.main(["--path", str(root), "--execute", "--limit", "1"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Execute mode: enabled" in output
    assert "Results: 1 success | 0 skipped | 0 failed" in output
    assert ("model", "http://127.0.0.1:8080/v1", "sakura-14b-qwen2.5-v1.0-q4km.gguf") in calls
    assert ("fw_model", "faster-whisper-large-v3") in calls
    assert ("run", 1, 1, 5, 1, "ja") in calls


def test_cli_execute_can_use_kotoba_python_transcriber(monkeypatch, tmp_path: Path, capsys):
    root = tmp_path / "root"
    audio = root / "work" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    calls = []

    def fake_preflight_model(base_url, model):
        calls.append(("model", base_url, model))

    def fake_preflight_kotoba():
        calls.append(("kotoba",))

    def fake_run_tasks(tasks, **kwargs):
        calls.append(
            (
                "run",
                kwargs["faster_whisper_config"].model,
                kwargs["faster_whisper_config"].device,
                kwargs["faster_whisper_config"].compute_type,
                kwargs["faster_whisper_config"].chunk_length,
                kwargs["faster_whisper_config"].condition_on_previous_text,
                kwargs["transcriber"].__name__,
            )
        )
        return [TaskResult(status="transcribed", reason="transcribe_complete", audio_path=tasks[0].audio_path, output_path=tasks[0].transcribe_srt_path)]

    monkeypatch.setattr(cli, "preflight_model", fake_preflight_model)
    monkeypatch.setattr(cli, "preflight_kotoba_faster_whisper", fake_preflight_kotoba)
    monkeypatch.setattr(cli, "run_tasks", fake_run_tasks)

    exit_code = cli.main(["--path", str(root), "--execute", "--limit", "1", "--transcribe_backend", "kotoba"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Transcription mode: kotoba python faster-whisper" in output
    assert ("kotoba",) in calls
    assert (
        "run",
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        "cuda",
        "float16",
        15,
        False,
        "run_kotoba_faster_whisper_transcription",
    ) in calls


def test_cli_execute_requires_positive_limit(monkeypatch, tmp_path: Path, capsys):
    root = tmp_path / "root"
    root.mkdir()

    exit_code = cli.main(["--path", str(root), "--execute", "--limit", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--limit must be >= 1" in captured.err


def test_effective_initial_prompt_defaults_to_asmr_prompt():
    result = _effective_initial_prompt(None, False)
    assert result == DEFAULT_ASMR_INITIAL_PROMPT
    assert "搾精" in result
    assert "射精" in result
    assert "精液" in result
    assert "おちんちん" in result
    assert "サキュバス" in result


def test_effective_initial_prompt_returns_none_when_disabled():
    assert _effective_initial_prompt(None, True) is None


def test_effective_initial_prompt_uses_custom_over_default():
    custom = "カスタムプロンプト"
    assert _effective_initial_prompt(custom, False) == custom


def test_effective_initial_prompt_custom_wins_over_no_asmr_prompt():
    custom = "カスタムプロンプト"
    assert _effective_initial_prompt(custom, True) == custom


def test_cli_execute_uses_asmr_prompt_by_default(monkeypatch, tmp_path: Path, capsys):
    root = tmp_path / "root"
    audio = root / "work" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    captured_config = {}

    monkeypatch.setattr(cli, "preflight_faster_whisper", lambda p: Path(p) / "fw.exe")
    monkeypatch.setattr(cli, "preflight_faster_whisper_model", lambda p, m: Path(p) / "whisper-faster" / m)
    monkeypatch.setattr(cli, "preflight_model", lambda *a, **k: None)

    def capture_run_tasks(tasks, **kwargs):
        captured_config["config"] = kwargs["faster_whisper_config"]
        return [cli.TaskResult(status="skipped", reason="existing_srt", audio_path=tasks[0].audio_path)]

    monkeypatch.setattr(cli, "run_tasks", capture_run_tasks)

    cli.main(["--path", str(root), "--execute", "--limit", "1"])
    cfg = captured_config["config"]
    assert cfg.initial_prompt == DEFAULT_ASMR_INITIAL_PROMPT
    assert cfg.reprompt is True


def test_cli_execute_disables_prompt_with_no_asmr_prompt(monkeypatch, tmp_path: Path, capsys):
    root = tmp_path / "root"
    audio = root / "work" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    captured_config = {}

    monkeypatch.setattr(cli, "preflight_faster_whisper", lambda p: Path(p) / "fw.exe")
    monkeypatch.setattr(cli, "preflight_faster_whisper_model", lambda p, m: Path(p) / "whisper-faster" / m)
    monkeypatch.setattr(cli, "preflight_model", lambda *a, **k: None)

    def capture_run_tasks(tasks, **kwargs):
        captured_config["config"] = kwargs["faster_whisper_config"]
        return [cli.TaskResult(status="skipped", reason="existing_srt", audio_path=tasks[0].audio_path)]

    monkeypatch.setattr(cli, "run_tasks", capture_run_tasks)

    cli.main(["--path", str(root), "--execute", "--no_asmr_prompt", "--limit", "1"])
    cfg = captured_config["config"]
    assert cfg.initial_prompt is None
    assert cfg.reprompt is False


def test_cli_execute_uses_custom_initial_prompt(monkeypatch, tmp_path: Path, capsys):
    root = tmp_path / "root"
    audio = root / "work" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    captured_config = {}

    monkeypatch.setattr(cli, "preflight_faster_whisper", lambda p: Path(p) / "fw.exe")
    monkeypatch.setattr(cli, "preflight_faster_whisper_model", lambda p, m: Path(p) / "whisper-faster" / m)
    monkeypatch.setattr(cli, "preflight_model", lambda *a, **k: None)

    def capture_run_tasks(tasks, **kwargs):
        captured_config["config"] = kwargs["faster_whisper_config"]
        return [cli.TaskResult(status="skipped", reason="existing_srt", audio_path=tasks[0].audio_path)]

    monkeypatch.setattr(cli, "run_tasks", capture_run_tasks)

    cli.main(["--path", str(root), "--execute", "--initial_prompt", "カスタムプロンプト", "--limit", "1"])
    assert captured_config["config"].initial_prompt == "カスタムプロンプト"


def test_cli_execute_reports_preflight_error_without_traceback(monkeypatch, tmp_path: Path, capsys):
    root = tmp_path / "root"
    audio = root / "work" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")

    monkeypatch.setattr(cli, "preflight_faster_whisper", lambda project_root: Path(project_root) / "fw.exe")

    def fail_model_preflight(project_root, model):
        raise FileNotFoundError("missing local faster-whisper model")

    monkeypatch.setattr(cli, "preflight_faster_whisper_model", fail_model_preflight)

    exit_code = cli.main(["--path", str(root), "--execute", "--limit", "1"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ERROR: missing local faster-whisper model" in captured.err
    assert "Traceback" not in captured.err


def test_cli_transcribe_only_skips_translation_preflight_and_translator(monkeypatch, tmp_path: Path, capsys):
    root = tmp_path / "root"
    audio = root / "work" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    seen = {}

    def fail_preflight_model(base_url, model):
        raise AssertionError("translation preflight must be skipped")

    def fake_run_tasks(tasks, **kwargs):
        seen["translator"] = kwargs["translator"]
        return [TaskResult(status="transcribed", reason="transcribe_complete", audio_path=tasks[0].audio_path, output_path=tasks[0].transcribe_srt_path)]

    monkeypatch.setattr(cli, "preflight_faster_whisper", lambda project_root: Path(project_root) / "fw.exe")
    monkeypatch.setattr(cli, "preflight_faster_whisper_model", lambda project_root, model: None)
    monkeypatch.setattr(cli, "preflight_model", fail_preflight_model)
    monkeypatch.setattr(cli, "run_tasks", fake_run_tasks)

    exit_code = cli.main(["--path", str(root), "--execute", "--transcribe_only"])

    assert exit_code == 0
    assert seen["translator"] is None
    output = capsys.readouterr().out
    assert "Preflight translation API: skipped" in output
    assert "Transcribed only (--transcribe_only): 1" in output


def _capture_config(monkeypatch, tmp_path: Path, extra_args: list[str]):
    root = tmp_path / "root"
    audio = root / "work" / "track.wav"
    audio.parent.mkdir(parents=True)
    audio.write_text("audio", encoding="utf-8")
    seen = {}

    def fake_run_tasks(tasks, **kwargs):
        seen["config"] = kwargs["faster_whisper_config"]
        return []

    monkeypatch.setattr(cli, "preflight_faster_whisper", lambda project_root: Path(project_root) / "fw.exe")
    monkeypatch.setattr(cli, "preflight_faster_whisper_model", lambda project_root, model: None)
    monkeypatch.setattr(cli, "preflight_model", lambda base_url, model: None)
    monkeypatch.setattr(cli, "run_tasks", fake_run_tasks)
    assert cli.main(["--path", str(root), "--execute", *extra_args]) == 0
    return seen["config"]


def test_fast_transcribe_uses_batched8_sentence_chunk20(monkeypatch, tmp_path: Path):
    config = _capture_config(monkeypatch, tmp_path, ["--fast_transcribe", "--transcribe_batch_size", "8"])

    assert (config.batched, config.batch_size, config.sentence, config.chunk_length, config.unmerged) == (True, 8, True, 20, False)


def test_default_transcribe_uses_batched12_sentence_chunk20(monkeypatch, tmp_path: Path):
    config = _capture_config(monkeypatch, tmp_path, [])

    assert (config.batched, config.batch_size, config.sentence, config.chunk_length, config.unmerged) == (True, 12, True, 20, False)


def test_fast_flag_preserves_default_batch12(monkeypatch, tmp_path: Path):
    config = _capture_config(monkeypatch, tmp_path, ["--fast_transcribe"])
    assert config.batched is True
    assert config.batch_size == 12


def test_batch_size_override_does_not_enable_sequential_batching(monkeypatch, tmp_path: Path):
    config = _capture_config(monkeypatch, tmp_path, ["--sequential_transcribe", "--transcribe_batch_size", "8"])
    assert (config.batched, config.batch_size, config.sentence, config.chunk_length) == (False, 8, False, None)


def test_cli_rejects_nonpositive_transcribe_batch_size(capsys):
    import pytest

    for size in ("0", "-1"):
        with pytest.raises(SystemExit) as error:
            cli.main(["--path", ".", "--transcribe_batch_size", size])
        assert error.value.code == 2
        assert "--transcribe_batch_size must be >= 1" in capsys.readouterr().err


def test_sequential_transcribe_explicitly_disables_batched_preset(monkeypatch, tmp_path: Path):
    config = _capture_config(monkeypatch, tmp_path, ["--sequential_transcribe"])

    assert (config.batched, config.sentence, config.chunk_length) == (False, False, None)


def test_transcription_presets_are_mutually_exclusive():
    import pytest

    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["--path", ".", "--fast_transcribe", "--sequential_transcribe"])
    assert error.value.code == 2
