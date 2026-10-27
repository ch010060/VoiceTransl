from pathlib import Path

import pytest

from scripts.asmr_transcribe_lib.kotoba_transformers import (
    DEFAULT_KOTOBA_TRANSFORMERS_MODEL,
    KotobaTransformersConfig,
    flatten_pipeline_chunks,
    format_srt_timestamp,
    run_kotoba_transformers_transcription,
)


class DummyPipeline:
    calls = []

    def __init__(self, output):
        self.output = output

    def __call__(self, audio_path, **kwargs):
        self.calls.append((audio_path, kwargs))
        return self.output


def test_flatten_pipeline_chunks_accepts_transformers_timestamp_shape():
    output = {
        "text": "ignored full text",
        "chunks": [
            {"timestamp": (0.0, 1.25), "text": "こんにちは。"},
            {"timestamp": [61.001, 62.5], "text": "耳、舐めてあげる。"},
            {"timestamp": (None, 70.0), "text": "skip missing start"},
            {"timestamp": (70.0, None), "text": "skip missing end"},
            {"timestamp": (71.0, 72.0), "text": ""},
        ],
    }

    assert flatten_pipeline_chunks(output) == [
        (0.0, 1.25, "こんにちは。"),
        (61.001, 62.5, "耳、舐めてあげる。"),
    ]


def test_kotoba_transformers_adapter_uses_cuda_defaults_and_writes_srt(tmp_path: Path):
    audio = tmp_path / "track.wav"
    target = tmp_path / "track-transcribe.srt"
    audio.write_text("audio", encoding="utf-8")
    pipeline = DummyPipeline(
        {
            "chunks": [
                {"timestamp": (0.0, 1.25), "text": "こんにちは。"},
                {"timestamp": (61.001, 62.5), "text": "耳、舐めてあげる。"},
            ]
        }
    )
    DummyPipeline.calls = []
    created = []

    def pipeline_factory(**kwargs):
        created.append(kwargs)
        return pipeline

    result = run_kotoba_transformers_transcription(
        audio_path=audio,
        transcribe_srt_path=target,
        config=KotobaTransformersConfig(),
        pipeline_factory=pipeline_factory,
    )

    assert result == target
    assert created == [
        {
            "model": DEFAULT_KOTOBA_TRANSFORMERS_MODEL,
            "torch_dtype": "float16",
            "device": "cuda:0",
            "model_kwargs": {"attn_implementation": "sdpa"},
            "batch_size": 16,
            "trust_remote_code": True,
            "punctuator": True,
        }
    ]
    assert DummyPipeline.calls == [
        (
            str(audio),
            {
                "chunk_length_s": 15,
                "return_timestamps": True,
                "generate_kwargs": {"language": "ja", "task": "transcribe"},
            },
        )
    ]
    assert target.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,250\n"
        "こんにちは。\n\n"
        "2\n"
        "00:01:01,001 --> 00:01:02,500\n"
        "耳、舐めてあげる。\n"
    )


def test_kotoba_transformers_adapter_can_force_standard_asr_task(tmp_path: Path):
    audio = tmp_path / "track.wav"
    target = tmp_path / "track-transcribe.srt"
    audio.write_text("audio", encoding="utf-8")
    pipeline = DummyPipeline({"chunks": []})
    created = []

    def pipeline_factory(**kwargs):
        created.append(kwargs)
        return pipeline

    run_kotoba_transformers_transcription(
        audio_path=audio,
        transcribe_srt_path=target,
        config=KotobaTransformersConfig(pipeline_task="automatic-speech-recognition"),
        pipeline_factory=pipeline_factory,
    )

    assert created[0]["task"] == "automatic-speech-recognition"


def test_kotoba_transformers_adapter_omits_chunk_length_when_zero(tmp_path: Path):
    audio = tmp_path / "track.wav"
    target = tmp_path / "track-transcribe.srt"
    audio.write_text("audio", encoding="utf-8")
    pipeline = DummyPipeline({"chunks": []})

    run_kotoba_transformers_transcription(
        audio_path=audio,
        transcribe_srt_path=target,
        config=KotobaTransformersConfig(chunk_length_s=0),
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    assert "chunk_length_s" not in DummyPipeline.calls[-1][1]


def test_kotoba_transformers_adapter_reports_missing_dependency(monkeypatch, tmp_path: Path):
    import scripts.asmr_transcribe_lib.kotoba_transformers as module

    audio = tmp_path / "track.wav"
    audio.write_text("audio", encoding="utf-8")

    def missing_import(name):
        if name in {"torch", "transformers"}:
            raise ImportError("missing")
        raise AssertionError(name)

    monkeypatch.setattr(module, "import_module", missing_import)

    with pytest.raises(RuntimeError, match="transformers.*torch"):
        run_kotoba_transformers_transcription(
            audio_path=audio,
            transcribe_srt_path=tmp_path / "track-transcribe.srt",
            config=KotobaTransformersConfig(),
            pipeline_factory=None,
        )


def test_format_srt_timestamp_rounds_to_milliseconds():
    assert format_srt_timestamp(3723.4567) == "01:02:03,457"
