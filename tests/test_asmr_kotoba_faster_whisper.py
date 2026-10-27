from pathlib import Path

import pytest

from scripts.asmr_transcribe_lib import kotoba_faster_whisper
from scripts.asmr_transcribe_lib.kotoba_faster_whisper import (
    KotobaFasterWhisperConfig,
    format_srt_timestamp,
    run_kotoba_faster_whisper_transcription,
)


class DummySegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class DummyWhisperModel:
    calls = []

    def __init__(self, model_size_or_path, **kwargs):
        self.calls.append(("init", model_size_or_path, kwargs))

    def transcribe(self, audio_path, **kwargs):
        self.calls.append(("transcribe", audio_path, kwargs))
        return [
            DummySegment(0.0, 1.25, "こんにちは。"),
            DummySegment(61.001, 62.5, "耳、舐めてあげる。"),
        ], object()


def test_format_srt_timestamp_uses_comma_milliseconds():
    assert format_srt_timestamp(3723.4567) == "01:02:03,457"


def test_kotoba_adapter_uses_official_recommended_defaults(tmp_path: Path):
    DummyWhisperModel.calls = []
    audio = tmp_path / "track.wav"
    target = tmp_path / "track-transcribe.srt"
    audio.write_text("audio", encoding="utf-8")

    result = run_kotoba_faster_whisper_transcription(
        audio_path=audio,
        transcribe_srt_path=target,
        project_root=tmp_path,
        config=KotobaFasterWhisperConfig(),
        whisper_model_cls=DummyWhisperModel,
    )

    assert result == target
    assert DummyWhisperModel.calls[0] == (
        "init",
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        {"device": "cuda", "compute_type": "float16"},
    )
    assert DummyWhisperModel.calls[1] == (
        "transcribe",
        str(audio),
        {"language": "ja", "chunk_length": 15, "condition_on_previous_text": False},
    )
    assert target.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,250\n"
        "こんにちは。\n\n"
        "2\n"
        "00:01:01,001 --> 00:01:02,500\n"
        "耳、舐めてあげる。\n"
    )


def test_kotoba_adapter_replaces_stale_output(tmp_path: Path):
    audio = tmp_path / "track.wav"
    target = tmp_path / "track-transcribe.srt"
    audio.write_text("audio", encoding="utf-8")
    target.write_text("stale", encoding="utf-8")

    run_kotoba_faster_whisper_transcription(
        audio_path=audio,
        transcribe_srt_path=target,
        project_root=tmp_path,
        config=KotobaFasterWhisperConfig(),
        whisper_model_cls=DummyWhisperModel,
    )

    assert "stale" not in target.read_text(encoding="utf-8")


def test_kotoba_adapter_reports_missing_dependency(monkeypatch, tmp_path: Path):
    audio = tmp_path / "track.wav"
    audio.write_text("audio", encoding="utf-8")

    def missing_import(name):
        if name == "faster_whisper":
            raise ImportError("missing")
        raise AssertionError(name)

    monkeypatch.setattr(kotoba_faster_whisper, "import_module", missing_import)

    with pytest.raises(RuntimeError, match="pip install faster-whisper"):
        run_kotoba_faster_whisper_transcription(
            audio_path=audio,
            transcribe_srt_path=tmp_path / "track-transcribe.srt",
            project_root=tmp_path,
            config=KotobaFasterWhisperConfig(),
            whisper_model_cls=None,
        )
