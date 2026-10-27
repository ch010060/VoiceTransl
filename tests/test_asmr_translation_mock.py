from pathlib import Path
import json

import httpx
import pytest

from scripts.asmr_transcribe_lib.openai_compatible import (
    ModelNotFoundError,
    OpenAICompatibleError,
    SAKURA_SYSTEM_PROMPT,
    SAKURA_USER_PREFIX,
    preflight_model,
)
from scripts.asmr_transcribe_lib.translation import translate_srt_to_simplified


def requested_lines(body) -> list[str]:
    assert body["messages"][0]["content"] == SAKURA_SYSTEM_PROMPT
    content = body["messages"][1]["content"]
    assert content.startswith(SAKURA_USER_PREFIX)
    return content[len(SAKURA_USER_PREFIX):].split("\n")


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://testserver")


def test_preflight_model_accepts_model_from_models_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "sakura-14b-qwen2.5-v1.0-q4km.gguf"}]})

    with make_client(handler) as client:
        preflight_model("http://127.0.0.1:8080/v1", "sakura-14b-qwen2.5-v1.0-q4km.gguf", client=client)


def test_preflight_model_reports_missing_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "other-model"}]})

    with make_client(handler) as client:
        with pytest.raises(ModelNotFoundError, match="sakura-14b"):
            preflight_model("http://127.0.0.1:8080/v1", "sakura-14b-qwen2.5-v1.0-q4km.gguf", client=client)


def test_preflight_model_reports_unsupported_models_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": []})

    with make_client(handler) as client:
        with pytest.raises(OpenAICompatibleError, match="models"):
            preflight_model("http://127.0.0.1:8080/v1", "model", client=client)


def test_translate_srt_to_simplified_preserves_index_and_timestamp(tmp_path: Path):
    source = tmp_path / "track-transcribe.srt"
    target = tmp_path / "track-translate.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n雨の音。\n",
        encoding="utf-8",
    )
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.read().decode("utf-8"))
        assert requested_lines(body) == ["こんにちは。", "雨の音。"]
        return httpx.Response(200, json={"choices": [{"message": {"content": '["你好。", "雨声。"]'}}]})

    with make_client(handler) as client:
        translate_srt_to_simplified(
            source,
            target,
            base_url="http://127.0.0.1:8080/v1",
            model="sakura-14b-qwen2.5-v1.0-q4km.gguf",
            client=client,
            max_blocks=30,
        )

    assert len(requests) == 1
    assert target.read_text(encoding="utf-8") == (
        "1\n00:00:01,000 --> 00:00:02,000\n你好。\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n雨声。\n"
    )


def test_translate_srt_retries_invalid_json_then_succeeds(tmp_path: Path):
    source = tmp_path / "track-transcribe.srt"
    target = tmp_path / "track-translate.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "你好。\n多余。"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '["你好。"]'}}]})

    with make_client(handler) as client:
        translate_srt_to_simplified(source, target, base_url="http://127.0.0.1:8080/v1", model="model", client=client)

    assert attempts["count"] == 2
    assert "你好" in target.read_text(encoding="utf-8")


def test_translate_srt_accepts_sakura_plain_line_response(tmp_path: Path):
    source = tmp_path / "track-transcribe.srt"
    target = tmp_path / "track-translate.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n雨の音。\n",
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "「你好。」\n「雨声。」"}}]})

    with make_client(handler) as client:
        translate_srt_to_simplified(source, target, base_url="http://127.0.0.1:8080/v1", model="model", client=client)

    assert target.read_text(encoding="utf-8") == (
        "1\n00:00:01,000 --> 00:00:02,000\n你好。\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n雨声。\n"
    )


def test_translate_srt_splits_chunk_when_batch_response_cannot_be_mapped(tmp_path: Path):
    source = tmp_path / "track-transcribe.srt"
    target = tmp_path / "track-translate.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n一。\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n二。\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\n三。\n",
        encoding="utf-8",
    )
    request_sizes = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode("utf-8"))
        requested = requested_lines(body)
        request_sizes.append(len(requested))
        if len(requested) > 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "合併成一行，無法對應。"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps([f"译{requested[0]}"], ensure_ascii=False)}}]})

    with make_client(handler) as client:
        translate_srt_to_simplified(
            source,
            target,
            base_url="http://127.0.0.1:8080/v1",
            model="model",
            client=client,
            max_blocks=3,
            max_retries=1,
        )

    assert request_sizes == [3, 1, 2, 1, 1]
    assert "译一" in target.read_text(encoding="utf-8")
    assert "译二" in target.read_text(encoding="utf-8")
    assert "译三" in target.read_text(encoding="utf-8")


def test_translate_srt_fails_safely_on_block_count_mismatch(tmp_path: Path):
    source = tmp_path / "track-transcribe.srt"
    target = tmp_path / "track-translate.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n雨の音。\n",
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode("utf-8"))
        requested = requested_lines(body)
        if len(requested) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '["只有一句。"]'}}]})

    with make_client(handler) as client:
        with pytest.raises(OpenAICompatibleError, match="count"):
            translate_srt_to_simplified(source, target, base_url="http://127.0.0.1:8080/v1", model="model", client=client, max_retries=2)

    assert not target.exists()


def test_translate_srt_caps_tokens_and_rejects_truncated_output(tmp_path: Path):
    source = tmp_path / "track-transcribe.srt"
    target = tmp_path / "track-translate.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n", encoding="utf-8")
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode("utf-8"))
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "啊啊啊啊"}, "finish_reason": "length"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '["你好。"]'}, "finish_reason": "stop"}]})

    with make_client(handler) as client:
        translate_srt_to_simplified(source, target, base_url="http://127.0.0.1:8080/v1", model="model", client=client)

    assert len(bodies) == 2
    assert all(0 < body["max_tokens"] <= 768 for body in bodies)
    assert bodies[1]["frequency_penalty"] > bodies[0]["frequency_penalty"]
    assert "你好" in target.read_text(encoding="utf-8")



def test_translate_srt_flattens_multiline_blocks_to_one_prompt_line(tmp_path: Path):
    source = tmp_path / "track-transcribe.srt"
    target = tmp_path / "track-translate.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\nお兄ちゃん。\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n雨の音。\n",
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode("utf-8"))
        assert requested_lines(body) == ["こんにちは。 お兄ちゃん。", "雨の音。"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "你好。哥哥。\n雨声。"}, "finish_reason": "stop"}]})

    with make_client(handler) as client:
        translate_srt_to_simplified(source, target, base_url="http://127.0.0.1:8080/v1", model="model", client=client)

    assert "你好。哥哥。" in target.read_text(encoding="utf-8")


def test_translate_srt_sends_repeated_lines_once(tmp_path: Path):
    source = tmp_path / "track-transcribe.srt"
    target = tmp_path / "track-translate.srt"
    texts = ["かきー。", "こきー。", "かきー。", "かきー。", "ね。", "こきー。"]
    source.write_text(
        "".join(f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},500\n{t}\n\n" for i, t in enumerate(texts, start=1)),
        encoding="utf-8",
    )
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        lines = requested_lines(json.loads(request.read().decode("utf-8")))
        sent.append(lines)
        reply = "\n".join(f"译{line}" for line in lines)
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}, "finish_reason": "stop"}]})

    with make_client(handler) as client:
        translate_srt_to_simplified(source, target, base_url="http://127.0.0.1:8080/v1", model="model", client=client)

    assert sent == [["かきー。", "こきー。", "ね。"]]
    out = target.read_text(encoding="utf-8")
    assert out.count("译かきー。") == 3
    assert out.count("译こきー。") == 2


def test_translate_srt_keeps_blank_blocks_without_sending_them(tmp_path: Path):
    source = tmp_path / "track-transcribe.srt"
    target = tmp_path / "track-translate.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nこんにちは。\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\n雨の音。\n",
        encoding="utf-8",
    )
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        lines = requested_lines(json.loads(request.read().decode("utf-8")))
        sent.append(lines)
        return httpx.Response(200, json={"choices": [{"message": {"content": "你好。\n雨声。"}, "finish_reason": "stop"}]})

    with make_client(handler) as client:
        translate_srt_to_simplified(source, target, base_url="http://127.0.0.1:8080/v1", model="model", client=client)

    assert sent == [["こんにちは。", "雨の音。"]]
    assert "你好。" in target.read_text(encoding="utf-8")
