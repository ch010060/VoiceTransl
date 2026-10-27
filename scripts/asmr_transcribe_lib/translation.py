from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import httpx

from .openai_compatible import DEFAULT_FREQUENCY_PENALTY, OpenAICompatibleError, chat_completion_json_array
from .srt_utils import parse_srt_file, replace_block_text, validate_same_timing, write_srt_file


def translate_srt_to_simplified(
    source_srt_path: Path | str,
    target_srt_path: Path | str,
    *,
    base_url: str,
    model: str,
    client: httpx.Client | None = None,
    max_blocks: int = 50,
    max_retries: int = 3,
) -> Path:
    source = Path(source_srt_path)
    target = Path(target_srt_path)
    blocks = parse_srt_file(source)
    source_texts = ["\n".join(block.text) for block in blocks]
    # Translate each distinct line once: long runs of repeated SFX lines (かきー。×N)
    # make Sakura lose count of lines, and repeats cost extra requests.
    # Blank blocks are kept as-is: an empty prompt line is dropped by the model and
    # would make every retry fail the line-count check.
    translations: dict[str, str] = {text: text for text in source_texts if not text.strip()}
    unique_texts = [text for text in dict.fromkeys(source_texts) if text not in translations]

    for chunk in _chunks(unique_texts, max_blocks):
        translations.update(
            zip(
                chunk,
                _translate_chunk_resilient(
                    chunk,
                    base_url=base_url,
                    model=model,
                    client=client,
                    max_retries=max_retries,
                ),
            )
        )

    translated_texts = [translations[text] for text in source_texts]
    replaced = replace_block_text(blocks, translated_texts)
    validate_same_timing(blocks, replaced)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.name}.tmp")
    try:
        write_srt_file(tmp_target, replaced)
        tmp_target.replace(target)
    except Exception:
        if tmp_target.exists():
            tmp_target.unlink()
        raise
    return target


def _translate_chunk_resilient(
    texts: Sequence[str],
    *,
    base_url: str,
    model: str,
    client: httpx.Client | None,
    max_retries: int,
) -> list[str]:
    try:
        return _translate_chunk_with_retry(
            texts,
            base_url=base_url,
            model=model,
            client=client,
            max_retries=max_retries,
        )
    except OpenAICompatibleError:
        if len(texts) <= 1:
            raise
        midpoint = len(texts) // 2
        return _translate_chunk_resilient(
            texts[:midpoint],
            base_url=base_url,
            model=model,
            client=client,
            max_retries=max_retries,
        ) + _translate_chunk_resilient(
            texts[midpoint:],
            base_url=base_url,
            model=model,
            client=client,
            max_retries=max_retries,
        )


def _translate_chunk_with_retry(
    texts: Sequence[str],
    *,
    base_url: str,
    model: str,
    client: httpx.Client | None,
    max_retries: int,
) -> list[str]:
    last_error: OpenAICompatibleError | None = None
    for attempt in range(max_retries):
        try:
            # Raise the penalty on each retry so a degenerate loop is not replayed verbatim.
            return chat_completion_json_array(
                base_url=base_url,
                model=model,
                texts=texts,
                client=client,
                frequency_penalty=DEFAULT_FREQUENCY_PENALTY + 0.1 * attempt,
            )
        except OpenAICompatibleError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _chunks(items: Sequence[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("max_blocks must be >= 1")
    return [list(items[index : index + size]) for index in range(0, len(items), size)]
