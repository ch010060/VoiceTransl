"""Read-only regression: SEcut variants must share the same base-track group."""
from scripts.asmr_transcribe_lib.planner import _content_stem_key, _same_content_stem
import pytest
from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan


def test_specific_rj01749122_regression():
    names = [
        "Track01_エロいやつ、探してる？",
        "Track01_エロいやつ、探してる？(SECut)",
        "Track08_夏のデートはラムネの味",
        "Track08_夏のデートはラムネの味(SECut)",
    ]
    keys = [_content_stem_key(n) for n in names]
    assert keys[0] == keys[1]
    assert keys[2] == keys[3]
    assert keys[0] != keys[2]


def test_secut_suffix_is_not_arbitrary_text_removal():
    assert _content_stem_key("Track01_SecurityCut") == "track01_securitycut"
    assert _content_stem_key("Track01_(SECut) Remix") != _content_stem_key("Track01")
    assert _content_stem_key("Track01 (SECut)") == _content_stem_key("Track01")
    assert _content_stem_key("Track01（SECut）") == _content_stem_key("Track01")
    assert _content_stem_key("Track01 (SEなし)") == _content_stem_key("Track01")


def test_different_track_numbers_remain_distinct():
    left = _content_stem_key("Track01_エロいやつ、探してる？(SECut)")
    right = _content_stem_key("Track08_夏のデートはラムネの味")
    assert not _same_content_stem(left, right)


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_four_variants_per_track_reuse_subtitles_once(tmp_path, mode):
    root = tmp_path / "library"
    originals = {}
    for number in ("01", "08"):
        for folder, suffix, tag in (("flac", ".flac", ""), ("wav", ".wav", ""),
                                    ("mp3", ".mp3", ""), ("SECut", ".wav", "(SECut)")):
            audio = root / "RJ01749122" / folder / ("Track" + number + "_Synthetic" + tag + suffix)
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"synthetic fixture " + number.encode())
            originals[audio] = audio.read_bytes()
    plan = build_execution_plan(root, mode=mode)
    assert plan.executable
    assert sorted(len(group.members) for group in plan.duplicate_groups) == [4, 4]
    assert sum(track.task.action == "process" for track in plan.tracks) == 2
    for group in plan.duplicate_groups:
        representative = plan.track_index[group.representative_track_id]
        assert representative.task.audio_path.suffix == ".flac"
    calls = []

    def asr(*, transcribe_srt_path, **kwargs):
        calls.append("asr")
        transcribe_srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nSynthetic.\n\n", encoding="utf-8")
        return transcribe_srt_path

    def translate(source, target):
        calls.append("translation")
        target.write_bytes(source.read_bytes())
        return target

    def convert(source, target):
        target.write_bytes(source.read_bytes())
        return target

    results = run_plan(plan, project_root=tmp_path, transcriber=asr,
        translator=translate if mode == "full" else None, opencc_converter=convert)
    assert calls.count("asr") == 2
    assert calls.count("translation") == (2 if mode == "full" else 0)
    assert sum(result.status == "copied" for result in results) == 6
    assert all(path.read_bytes() == content for path, content in originals.items())
    for track in plan.tracks:
        output = track.task.final_srt_path if mode == "full" else track.task.transcribe_srt_path
        assert output.is_file()
