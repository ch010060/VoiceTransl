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


DEFAULT_MAX_TOKENS_CAP = 768
DEFAULT_FREQUENCY_PENALTY = 0.1

# Sakura v1.0 native prompt (see GalTransl/Backend/Prompts.py). Sakura is trained on
# newline-separated text, not JSON, so one subtitle per line maps back reliably.
SAKURA_SYSTEM_PROMPT = (
    "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，注意不要擅自添加原文中没有的代词，也不要擅自增加或减少换行。"
)
SAKURA_USER_PREFIX = "将下面的日文文本翻译成中文："


def to_prompt_lines(texts: Sequence[str]) -> list[str]:
    # Multi-line SRT blocks are flattened so each block is exactly one prompt line.
    return [" ".join(part.strip() for part in text.splitlines() if part.strip()) for text in texts]


def estimate_max_tokens(texts: Sequence[str], *, cap: int = DEFAULT_MAX_TOKENS_CAP) -> int:
    # Without a cap llama-server generates until the slot is full, then context-shifts
    # (dropping the system prompt) and loops forever when Sakura degenerates.
    input_chars = sum(len(line) + 1 for line in to_prompt_lines(texts))
    return max(64, min(cap, 64 + int(input_chars * 1.5)))


def chat_completion_json_array(
    *,
    base_url: str,
    model: str,
    texts: Sequence[str],
    client: httpx.Client | None = None,
    timeout: float = 120.0,
    max_tokens: int | None = None,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
) -> list[str]:
    owns_client = client is None
    active_client = client or httpx.Client(timeout=timeout)
    try:
        response = active_client.post(
            endpoint_url(base_url, "chat/completions"),
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SAKURA_SYSTEM_PROMPT},
                    {"role": "user", "content": SAKURA_USER_PREFIX + "\n".join(to_prompt_lines(texts))},
                ],
                # Sakura-recommended sampling; frequency_penalty breaks repetition loops.
                "temperature": 0.1,
                "top_p": 0.3,
                "frequency_penalty": frequency_penalty,
                "max_tokens": max_tokens if max_tokens is not None else estimate_max_tokens(texts),
            },
        )
        response.raise_for_status()
        payload = response.json()
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenAICompatibleError("unsupported chat/completions response shape") from exc
        if choice.get("finish_reason") == "length":
            raise OpenAICompatibleError("chat/completions output truncated at max_tokens (likely repetition loop)")
        return parse_translation_content(content, expected_count=len(texts))
    except httpx.HTTPError as exc:
        raise OpenAICompatibleError(f"chat/completions request failed: {exc}") from exc
    finally:
        if owns_client:
            active_client.close()


def parse_translation_content(content: str, *, expected_count: int) -> list[str]:
    if not content.lstrip().startswith("["):
        lines = _parse_plain_line_translations(content, expected_count=expected_count)
        if lines is None:
            raise OpenAICompatibleError(f"translation count mismatch: expected {expected_count} lines")
        return lines
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
