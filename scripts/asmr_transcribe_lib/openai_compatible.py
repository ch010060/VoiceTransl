from __future__ import annotations

import json
from collections.abc import Sequence

import httpx


class OpenAICompatibleError(RuntimeError):
    pass


class ModelNotFoundError(OpenAICompatibleError):
    pass


def endpoint_url(base_url: str, suffix: str) -> str:
    return f"{base_url.rstrip('/')}/{suffix.lstrip('/')}"


def preflight_model(base_url: str, model: str, *, client: httpx.Client | None = None, timeout: float = 30.0) -> None:
    owns_client = client is None
    active_client = client or httpx.Client(timeout=timeout)
    try:
        response = active_client.get(endpoint_url(base_url, "models"))
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list):
            raise OpenAICompatibleError("unsupported /models response shape: expected top-level data list")
        model_ids = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                model_ids.append(item["id"])
        if model not in model_ids:
            raise ModelNotFoundError(f"model not found in /models: {model}")
    except httpx.HTTPError as exc:
        raise OpenAICompatibleError(f"models endpoint request failed: {exc}") from exc
    finally:
        if owns_client:
            active_client.close()


def chat_completion_json_array(
    *,
    base_url: str,
    model: str,
    texts: Sequence[str],
    client: httpx.Client | None = None,
    timeout: float = 120.0,
) -> list[str]:
    owns_client = client is None
    active_client = client or httpx.Client(timeout=timeout)
    try:
        response = active_client.post(
            endpoint_url(base_url, "chat/completions"),
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是字幕翻譯器。請把輸入的日文或來源語字幕翻譯成簡體中文。"
                            "只輸出 JSON array of strings，不要輸出 markdown，不要改變順序。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(list(texts), ensure_ascii=False),
                    },
                ],
                "temperature": 0,
            },
        )
        response.raise_for_status()
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenAICompatibleError("unsupported chat/completions response shape") from exc
        return parse_translation_content(content, expected_count=len(texts))
    except httpx.HTTPError as exc:
        raise OpenAICompatibleError(f"chat/completions request failed: {exc}") from exc
    finally:
        if owns_client:
            active_client.close()


def parse_translation_content(content: str, *, expected_count: int) -> list[str]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        parsed = _parse_plain_line_translations(content, expected_count=expected_count)
        if parsed is None:
            raise OpenAICompatibleError(f"chat/completions content is not valid JSON: {exc}") from exc
        return parsed

    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise OpenAICompatibleError("chat/completions content must be a JSON array of strings")
    if len(parsed) != expected_count:
        raise OpenAICompatibleError(f"translation count mismatch: expected {expected_count}, got {len(parsed)}")
    return parsed


def _parse_plain_line_translations(content: str, *, expected_count: int) -> list[str] | None:
    lines = [_strip_translation_quotes(line.strip()) for line in content.splitlines() if line.strip()]
    if len(lines) != expected_count:
        return None
    return lines


def _strip_translation_quotes(text: str) -> str:
    pairs = [("「", "」"), ("『", "』"), ('"', '"'), ("'", "'")]
    stripped = text.strip()
    changed = True
    while changed and len(stripped) >= 2:
        changed = False
        for left, right in pairs:
            if stripped.startswith(left) and stripped.endswith(right):
                stripped = stripped[len(left) : -len(right)].strip()
                changed = True
                break
    return stripped
