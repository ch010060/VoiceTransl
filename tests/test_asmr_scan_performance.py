"""Scan-local reuse must preserve the uncached plan and fresh trust boundaries."""
from pathlib import Path
import os
import subprocess

import pytest

from scripts.asmr_transcribe_lib import gui_plan, planner
from scripts.asmr_transcribe_lib.gui_plan_io import encode_plan, decode_plan, PlanTransportError


def library(tmp_path):
    root = tmp_path / "library"
    for work in ("A", "B"):
        for name in ("wav/Track01 Voice.wav", "mp3/Track01 Voice.mp3", "SE/Track01 Voice(SECut).wav", "Track02 Other.wav"):
            path = root / work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic " + work.encode())
    return root


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
@pytest.mark.parametrize("existing", [False, True])
def test_cached_and_uncached_plans_are_identical(tmp_path, mode, existing):
    root = library(tmp_path)
    if existing:
        (root / "A/wav/Track01 Voice.VtT").write_text("WEBVTT\n", encoding="utf-8")
        (root / "B/Track02 Other.ASS").mkdir()
    plain = gui_plan.build_execution_plan.__wrapped__(root, mode=mode, skip_existing_subtitles=True)
    cached = gui_plan.build_execution_plan(root, mode=mode, skip_existing_subtitles=True)
    assert plain == cached


def test_group_matching_not_repeated_during_gui_plan(tmp_path, monkeypatch):
    root = library(tmp_path)
    calls = []
    original = planner._same_content_task

    def compare(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(planner, "_same_content_task", compare)
    gui_plan.build_execution_plan(root)
    assert len(calls) == 2 * 6  # Two Works, four audio paths, one all-pairs pass.
    calls.clear()
    planner.discover_audio_tasks(root)
    assert len(calls) == 2 * 6  # CLI still independently discovers and matches.


def test_subtitle_directory_enumerated_once_per_scan(tmp_path, monkeypatch):
    root = tmp_path / "library"
    work = root / "Work"
    work.mkdir(parents=True)
    for index in range(30):
        (work / f"Track{index:02}.wav").write_bytes(b"synthetic")
    (work / "Track00.vtt").write_text("WEBVTT\n", encoding="utf-8")
    calls = []
    original = Path.iterdir

    def iterdir(path):
        if path == work:
            calls.append(1)
        return original(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    gui_plan.build_execution_plan(root, skip_existing_subtitles=True)
    assert len(calls) <= 1


def test_new_scan_refreshes_subtitle_and_audio_identity(tmp_path):
    root = library(tmp_path)
    before = gui_plan.build_execution_plan(root, skip_existing_subtitles=True)
    audio = root / "B/Track02 Other.wav"
    audio.write_bytes(b"modified synthetic audio")
    srt = audio.with_suffix(".srt")
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nNew.\n\n", encoding="utf-8")
    after = gui_plan.build_execution_plan(root, skip_existing_subtitles=True)
    tid = next(t.track_id for t in after.tracks if t.task.audio_path == audio)
    assert after.track_index[tid].source_stamp != before.track_index[tid].source_stamp
    assert after.track_index[tid].task.reason == "existing_subtitle_skip"
    srt.unlink()
    final = gui_plan.build_execution_plan(root, skip_existing_subtitles=True)
    assert final.track_index[tid].task.action == "process"


@pytest.mark.parametrize("change", ["add", "remove"])
def test_transport_membership_is_fresh_after_prior_encode(tmp_path, change):
    root = library(tmp_path)
    plan = gui_plan.build_execution_plan(root)
    document = encode_plan(plan)
    if change == "add":
        (root / "B/new.wav").write_bytes(b"new")
    else:
        (root / "B/Track02 Other.wav").unlink()
    with pytest.raises(PlanTransportError):
        decode_plan(document)


def test_subtitle_hash_is_not_reused_across_phases(tmp_path):
    root = library(tmp_path)
    subtitle = root / "B/Track02 Other.srt"
    subtitle.write_text("first", encoding="utf-8")
    plan = gui_plan.build_execution_plan(root, skip_existing_subtitles=True)
    previous = dict(plan.inventory)[subtitle]
    subtitle.write_text("other", encoding="utf-8")
    assert gui_plan.file_stamp(subtitle) != previous


def test_resolver_matches_pathlib_for_existing_and_missing_paths(tmp_path):
    from scripts.asmr_transcribe_lib.scan_cache import scan_phase, resolved_path
    root = library(tmp_path)
    paths = [root, root / "A/wav/Track01 Voice.wav", root / "A/missing/subtitle.srt", root / "A/missing/../outside.srt"]

    @scan_phase
    def resolve():
        return [resolved_path(path) for path in paths]

    assert resolve() == [path.resolve() for path in paths]


def junction(link, target):
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                   capture_output=True, check=True)
    assert link.is_junction()


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point identity")
def test_windows_junction_alias_and_missing_leaf_use_real_identity(tmp_path):
    from scripts.asmr_transcribe_lib.scan_cache import scan_phase, resolved_path
    target = tmp_path / "target"
    target.mkdir()
    (target / "voice.wav").write_bytes(b"synthetic")
    alias = tmp_path / "alias"
    junction(alias, target)

    @scan_phase
    def resolve():
        return [resolved_path(alias / name) for name in ("voice.wav", "missing.srt", "nested/missing.srt")]

    assert resolve() == [(alias / name).resolve() for name in ("voice.wav", "missing.srt", "nested/missing.srt")]
    assert resolve()[1] == target / "missing.srt"


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point identity")
def test_retargeted_junction_is_fresh_in_next_phase(tmp_path):
    from scripts.asmr_transcribe_lib.scan_cache import scan_phase, resolved_path
    first, second, alias = tmp_path / "first", tmp_path / "second", tmp_path / "alias"
    first.mkdir()
    second.mkdir()
    junction(alias, first)

    @scan_phase
    def resolve():
        return resolved_path(alias / "missing.srt")

    assert resolve() == first / "missing.srt"
    alias.rmdir()
    junction(alias, second)
    assert resolve() == second / "missing.srt"


@pytest.mark.skipif(os.name != "nt", reason="Windows namespace aliases")
def test_junction_alias_output_collisions_remain_rejected(tmp_path):
    root = tmp_path / "library"
    work = root / "Work"
    work.mkdir(parents=True)
    (work / "voice.wav").write_bytes(b"synthetic")
    junction(root / "Alias", work)
    plan = gui_plan.build_execution_plan(root)
    assert len(plan.tracks) == 2
    assert "OUTPUT_NAMESPACE_COLLISION" in {issue.code for issue in plan.issues}
    assert plan == gui_plan.build_execution_plan.__wrapped__(root)


@pytest.mark.parametrize("change", ["modify", "delete"])
def test_start_inventory_validation_is_fresh_after_scan(tmp_path, change):
    root = library(tmp_path)
    plan = gui_plan.build_execution_plan(root)
    encode_plan(plan)
    path = root / "B/Track02 Other.wav"
    if change == "modify":
        path.write_bytes(b"changed after scan")
    else:
        path.unlink()
    with pytest.raises(gui_plan.PlanRejected):
        plan.assert_ready()
