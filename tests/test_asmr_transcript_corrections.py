from pathlib import Path

from scripts.asmr_transcribe_lib.srt_utils import parse_srt_file
from scripts.asmr_transcribe_lib.transcript_corrections import (
    CorrectionRule,
    apply_default_traditional_chinese_corrections,
    apply_transcript_corrections,
    load_correction_rules,
)


def test_apply_transcript_corrections_uses_context_and_preserves_timing(tmp_path: Path):
    source = tmp_path / "track-transcribe.srt"
    target = tmp_path / "track-transcribe-corrected.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nクロエも、立派なサキュバスなのだから、ちゃんと作成もできるようにならねばな。\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n資料を作成してください。\n",
        encoding="utf-8",
    )
    rules = [
        CorrectionRule(
            wrong="作成",
            correct="搾精",
            include_any=("サキュバス",),
            note="adult context only",
        )
    ]

    result = apply_transcript_corrections(source, target, rules)

    blocks = parse_srt_file(result)
    assert blocks[0].timestamp == "00:00:01,000 --> 00:00:02,000"
    assert "搾精" in blocks[0].text[0]
    assert "資料を作成" in blocks[1].text[0]


def test_load_correction_rules_from_project_config():
    rules = load_correction_rules(Path("config/asmr_transcript_corrections.json"))

    assert any(rule.wrong == "作成" and rule.correct == "搾精" for rule in rules)
    assert any(rule.wrong == "気持ちはない" and rule.correct == "気持ち悪い" for rule in rules)
    assert any(rule.wrong == "こう都合" and rule.correct == "好都合" for rule in rules)
    assert any(rule.wrong == "射線" and rule.correct == "射精" for rule in rules)


def test_default_adult_asrm_corrections_fix_common_whisper_errors(tmp_path: Path):
    source = tmp_path / "track-transcribe.srt"
    target = tmp_path / "track-transcribe-corrected.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nじゃあこう都合じゃない。\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nキスして、明日、入れてきていいわよ。\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\n早く射線しちゃいなさい。\n\n"
        "4\n00:00:07,000 --> 00:00:08,000\n湿が汚れないよ、水液。お手で受け止めてあげるから。\n\n"
        "5\n00:00:09,000 --> 00:00:10,000\n8日、完全に勃起してる。\n",
        encoding="utf-8",
    )
    rules = load_correction_rules(Path("config/asmr_transcript_corrections.json"))

    result = apply_transcript_corrections(source, target, rules)

    blocks = parse_srt_file(result)
    assert blocks[0].text == ("じゃあ好都合じゃない。",)
    assert blocks[1].text == ("キスして、舌、入れてきていいわよ。",)
    assert blocks[2].text == ("早く射精しちゃいなさい。",)
    assert blocks[3].text == ("シーツが汚れないよ、精液。お手で受け止めてあげるから。",)
    assert blocks[4].text == ("おちんちん、完全に勃起してる。",)


def test_default_succubus_context_corrections_fix_sample_errors(tmp_path: Path):
    source = tmp_path / "succubus-transcribe.srt"
    target = tmp_path / "succubus-transcribe-corrected.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nクロエも、立派なサキバスなのだから、\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nちゃんと作成もできるようにならねばな。\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nこいつの正義も、いただき放題、だしな。\n\n"
        "4\n00:00:07,000 --> 00:00:08,000\nティファニー様からいただけるせいで十分ですのね。\n\n"
        "5\n00:00:09,000 --> 00:00:10,000\n五感への攻撃だけは良してやれ。\n",
        encoding="utf-8",
    )
    rules = load_correction_rules(Path("config/asmr_transcript_corrections.json"))

    result = apply_transcript_corrections(source, target, rules)

    blocks = parse_srt_file(result)
    assert blocks[0].text == ("クロエも、立派なサキュバスなのだから、",)
    assert blocks[1].text == ("ちゃんと搾精もできるようにならねばな。",)
    assert blocks[2].text == ("こいつの精気も、いただき放題、だしな。",)
    assert blocks[3].text == ("ティファニー様からいただける精で十分ですので。",)
    assert blocks[4].text == ("五感への攻撃だけはよしてやれ。",)


def test_apply_default_traditional_chinese_corrections_preserves_timing(tmp_path: Path):
    root = tmp_path / "project"
    config_dir = root / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "asmr_traditional_chinese_corrections.json").write_text(
        """
{
  "rules": [
    {"wrong": "想象", "correct": "想像"},
    {"wrong": "好惡心", "correct": "好噁心"}
  ]
}
""",
        encoding="utf-8",
    )
    srt = root / "track-opencc.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n你在想象什麼？好惡心。\n",
        encoding="utf-8",
    )

    result = apply_default_traditional_chinese_corrections(srt, project_root=root)

    [block] = parse_srt_file(result)
    assert block.timestamp == "00:00:01,000 --> 00:00:02,000"
    assert block.text == ("你在想像什麼？好噁心。",)
