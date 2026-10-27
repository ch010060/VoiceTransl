from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from .srt_utils import SubtitleBlock, parse_srt_file, write_srt_file

DEFAULT_CORRECTION_CONFIG = Path("config") / "asmr_transcript_corrections.json"
DEFAULT_TRADITIONAL_CHINESE_CORRECTION_CONFIG = Path("config") / "asmr_traditional_chinese_corrections.json"


@dataclass(frozen=True)
class CorrectionRule:
    wrong: str
    correct: str
    include_any: tuple[str, ...] = ()
    exclude_any: tuple[str, ...] = ()
    note: str = ""


def load_correction_rules(path: Path | str) -> list[CorrectionRule]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw_rules = payload.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError("transcript correction config must contain a rules list")

    rules: list[CorrectionRule] = []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            raise ValueError("each transcript correction rule must be an object")
        wrong = raw.get("wrong")
        correct = raw.get("correct")
        if not isinstance(wrong, str) or not isinstance(correct, str):
            raise ValueError("each transcript correction rule needs string wrong/correct")
        rules.append(
            CorrectionRule(
                wrong=wrong,
                correct=correct,
                include_any=tuple(_string_list(raw.get("include_any", []), "include_any")),
                exclude_any=tuple(_string_list(raw.get("exclude_any", []), "exclude_any")),
                note=str(raw.get("note", "")),
            )
        )
    return rules


def apply_default_transcript_corrections(
    srt_path: Path | str,
    *,
    project_root: Path | str,
) -> Path:
    config_path = Path(project_root) / DEFAULT_CORRECTION_CONFIG
    if not config_path.exists():
        return Path(srt_path)
    return apply_transcript_corrections(srt_path, srt_path, load_correction_rules(config_path))


def apply_default_traditional_chinese_corrections(
    srt_path: Path | str,
    *,
    project_root: Path | str,
) -> Path:
    config_path = Path(project_root) / DEFAULT_TRADITIONAL_CHINESE_CORRECTION_CONFIG
    if not config_path.exists():
        return Path(srt_path)
    return apply_transcript_corrections(srt_path, srt_path, load_correction_rules(config_path))


def apply_transcript_corrections(
    source_srt_path: Path | str,
    target_srt_path: Path | str,
    rules: Sequence[CorrectionRule],
) -> Path:
    source = Path(source_srt_path)
    target = Path(target_srt_path)
    blocks = parse_srt_file(source)
    corrected_blocks = [_correct_block(block, rules) for block in blocks]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.name}.tmp")
    try:
        write_srt_file(tmp_target, corrected_blocks)
        tmp_target.replace(target)
    except Exception:
        if tmp_target.exists():
            tmp_target.unlink()
        raise
    return target


def _correct_block(block: SubtitleBlock, rules: Sequence[CorrectionRule]) -> SubtitleBlock:
    text_lines = list(block.text)
    block_text = "\n".join(text_lines)
    for rule in rules:
        if not _rule_applies(block_text, rule):
            continue
        text_lines = [line.replace(rule.wrong, rule.correct) for line in text_lines]
        block_text = "\n".join(text_lines)
    return SubtitleBlock(index=block.index, timestamp=block.timestamp, text=tuple(text_lines))


def _rule_applies(text: str, rule: CorrectionRule) -> bool:
    if rule.wrong not in text:
        return False
    if rule.include_any and not any(token in text for token in rule.include_any):
        return False
    if rule.exclude_any and any(token in text for token in rule.exclude_any):
        return False
    return True


def _string_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return value
