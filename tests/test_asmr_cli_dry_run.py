import subprocess
import sys
from pathlib import Path


def test_cli_dry_run_lists_queued_skipped_and_copy_actions(tmp_path: Path):
    root = tmp_path / "root"
    (root / "work" / "01").mkdir(parents=True)
    (root / "work" / "01" / "queued.wav").write_text("audio", encoding="utf-8")
    (root / "work" / "01" / "done.flac").write_text("audio", encoding="utf-8")
    (root / "work" / "01" / "done.srt").write_text("existing", encoding="utf-8")
    (root / "work" / "01" / "copy.mp3").write_text("audio", encoding="utf-8")
    (root / "work" / "01" / "copy.vtt").write_text("WEBVTT", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/asmr_transcribe_srt_batch.py",
            "--path",
            str(root),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Preflight skipped: dry-run mode" in result.stdout
    assert "Transcription mode: local faster-whisper" in result.stdout
    assert "Faster-Whisper param file: whisper-faster/param.txt" in result.stdout
    assert "Faster-Whisper executable:" in result.stdout
    assert "Whisper-Faster/whisper-faster.exe" in result.stdout
    assert "Whisper-Faster/whisper-faster.exe" in result.stdout
    assert "Transcription model: faster-whisper-large-v3\n" in result.stdout
    assert "Transcription API" not in result.stdout
    assert "Audio:" in result.stdout
    assert "queued" in result.stdout
    assert "skip" in result.stdout
    assert "copy-subtitle" in result.stdout
    assert "http://127.0.0.1:8080/v1" in result.stdout


def test_cli_help_does_not_offer_external_transcription_api():
    result = subprocess.run(
        [sys.executable, "scripts/asmr_transcribe_srt_batch.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--transcribe_api" not in result.stdout
    assert "--faster_whisper_param" in result.stdout


