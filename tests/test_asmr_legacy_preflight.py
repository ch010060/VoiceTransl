"""Real-library rejection shapes, with synthetic files and no backend services."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.asmr_gui.diagnostics import plan_rejection_diagnostic
from scripts.asmr_transcribe_lib.gui_plan import (
    PlanRejected, build_execution_plan, make_provenance,
)
from scripts.asmr_transcribe_lib.gui_plan_io import decode_plan, encode_plan, PlanTransportError
from scripts.asmr_transcribe_lib.gui_results import read_json, reconcile_run, resume_provenance
from test_asmr_shared_output import library, execute, rehash, SRT
from test_asmr_gui_existing_subtitles import inventory


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
@pytest.mark.parametrize("formats", [("flac", "mp3"), ("flac", "wav", "mp3"), ("wav", "mp3")])
def test_existing_source_same_stem_has_one_writer_but_still_requires_proof(tmp_path, monkeypatch, mode, formats):
    root = library(tmp_path, formats)
    source = root / "RJ001/01_normal/Track01.srt"
    source.write_text(SRT, encoding="utf-8")
    before = inventory(root)
    legacy = build_execution_plan(root, mode=mode, skip_existing_subtitles=True)
    assert {i.code for i in legacy.issues} == {"UNVERIFIED_LEGACY"}
    assert len(legacy.issues) == 1 and legacy.issues[0].paths == (source,)
    aliases = [t for t in legacy.tracks if t.shared_output_owner]
    assert len(aliases) == len(formats) - 1
    assert decode_plan(encode_plan(legacy)) == legacy
    with pytest.raises(PlanRejected, match="UNVERIFIED_LEGACY"):
        legacy.assert_ready()
    assert inventory(root) == before

    # Only the caller's known completed-stage proof permits a real copy.
    source_track = next(t for t in legacy.tracks if t.task.final_srt_path == source)
    proof = make_provenance(legacy, source_track.track_id, source, "final")
    plan = build_execution_plan(root, mode=mode, skip_existing_subtitles=True, provenance=[proof])
    assert plan.executable and plan.duplicate_groups == legacy.duplicate_groups
    assert [t.track_id for t in plan.tracks] == [t.track_id for t in legacy.tracks]
    role = "transcription" if mode == "transcribe_only" else "final"
    assert sum(u.access == "write" and u.role == role for u in plan.output_ownership) == 1
    replay = decode_plan(encode_plan(plan))
    store, outcomes, result, calls, publishes = execute(replay, tmp_path / "run", monkeypatch)
    assert not calls and sum(publishes.values()) == 1
    assert Counter(o.status for o in outcomes) == {"skipped": 1, "copied": len(formats)}
    assert result["outcome"] == "completed"
    assert reconcile_run(store.run_dir, store.run_id, replay.plan_hash)["complete"]
    for track in aliases:
        receipt = read_json(store.run_dir / "receipts" / (track.track_id + ".json"))
        assert receipt["shared_output_owner"] == track.shared_output_owner
        assert receipt["shared_output_receipt"]["receipt_ref"].endswith(track.shared_output_owner + ".json")
    skipped = read_json(store.run_dir / "receipts" / (source_track.track_id + ".json"))
    assert skipped["provenance"] is None and "unverified_legacy" in skipped["warnings"]
    assert all(p.track_id != source_track.track_id for p in resume_provenance(store.run_dir, store.run_id, replay.plan_hash))
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == stamp for p, stamp in before.items())


@pytest.mark.parametrize("role", ["transcribe", "translate", "opencc"])
@pytest.mark.parametrize("temporary", [False, True])
def test_unknown_intermediate_and_tmp_fail_closed_without_modification(tmp_path, role, temporary):
    root = library(tmp_path, works=("RJ001", "RJ002"))
    path = root / "RJ001/01_normal" / ("Track01-" + role + ".srt" + (".tmp" if temporary else ""))
    path.write_text(SRT, encoding="utf-8")
    before = inventory(root)
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    assert any(i.code == "UNVERIFIED_LEGACY" and path in i.paths for i in plan.issues)
    with pytest.raises(PlanRejected, match="UNVERIFIED_LEGACY"):
        plan.assert_ready()
    assert inventory(root) == before
    # A different Work can still be explicitly selected and planned safely.
    selected = next(w.work_id for w in plan.works if w.relative_path == "RJ002")
    clean = build_execution_plan(root, skip_existing_subtitles=True, selected_work_ids=[selected])
    assert clean.executable and decode_plan(encode_plan(clean)) == clean


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
@pytest.mark.parametrize("existing_source", [False, True])
def test_copy_target_stale_tmp_is_not_completion_or_reusable_output(tmp_path, mode, existing_source):
    root = library(tmp_path)
    if existing_source:
        (root / "RJ001/01_normal/Track01.srt").write_text(SRT, encoding="utf-8")
    name = "Track01(SEcut)" + ("-transcribe" if mode == "transcribe_only" else "") + ".srt.tmp"
    stale = root / "RJ001/02_variant" / name
    stale.write_text(SRT, encoding="utf-8")
    before = inventory(root)
    plan = build_execution_plan(root, mode=mode, skip_existing_subtitles=True)
    assert any(i.code == "UNVERIFIED_LEGACY" and stale in i.paths for i in plan.issues)
    with pytest.raises(PlanRejected):
        plan.assert_ready()
    assert inventory(root) == before


def test_mixed_work_keeps_local_final_skip_unverified_and_does_not_copy_it(tmp_path, monkeypatch):
    root = library(tmp_path)
    legacy = root / "RJ001/02_variant/Track01(SEcut).srt"
    legacy.write_text(SRT, encoding="utf-8")
    fresh = root / "RJ001/03_new/Track99.wav"
    fresh.parent.mkdir()
    fresh.write_bytes(b"SYNTHETIC_NEW_AUDIO")
    before = inventory(root)
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    assert {i.code for i in plan.issues} == {"UNVERIFIED_LEGACY"}
    assert all(t.verification == "unverified_legacy" and t.task.action == "skip"
               for t in plan.tracks if t.task.audio_path.parent == legacy.parent)
    assert next(t for t in plan.tracks if t.task.audio_path == fresh).task.action == "process"
    with pytest.raises(PlanRejected):
        execute(plan, tmp_path / "run", monkeypatch)
    assert inventory(root) == before


def test_existing_copy_alias_forgery_is_rejected(tmp_path):
    root = library(tmp_path)
    (root / "RJ001/01_normal/Track01.srt").write_text(SRT, encoding="utf-8")
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    reader = next(t for t in plan.tracks if t.shared_output_owner)
    forged = rehash(replace(plan, tracks=tuple(
        replace(t, shared_output_owner=None) if t.track_id == reader.track_id else t for t in plan.tracks)))
    with pytest.raises(PlanTransportError, match="shared output ownership"):
        encode_plan(forged)


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_existing_copy_owner_failure_does_not_certify_alias(tmp_path, monkeypatch, mode):
    root = library(tmp_path)
    source = root / "RJ001/01_normal/Track01.srt"
    source.write_text(SRT, encoding="utf-8")
    legacy = build_execution_plan(root, mode=mode, skip_existing_subtitles=True)
    owner = next(t for t in legacy.tracks if t.task.final_srt_path == source)
    proof = make_provenance(legacy, owner.track_id, source, "final")
    plan = build_execution_plan(root, mode=mode, skip_existing_subtitles=True, provenance=[proof])
    before = inventory(root)
    store, outcomes, result, calls, publishes = execute(plan, tmp_path / "run", monkeypatch, fail_copy=True)
    assert Counter(o.status for o in outcomes) == {"skipped": 1, "failed": 1, "not_started": 1}
    assert not calls and sum(publishes.values()) == 1
    assert result["outcome"] == "completed_with_errors"
    assert not resume_provenance(store.run_dir, store.run_id, plan.plan_hash)
    assert inventory(root) == before


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_local_final_skip_preserves_old_intermediates_while_other_work_runs(tmp_path, mode):
    from test_asmr_gui_existing_subtitles import make_library, execute as execute_mixed
    root = make_library(tmp_path, ("Old/Track01.wav", "New/Track99.wav"))
    audio = root / "Old/Track01.wav"
    final = audio.with_suffix(".srt")
    final.write_text(SRT, encoding="utf-8")
    for suffix in ("-transcribe.srt", "-translate.srt", "-opencc.srt", ".srt.tmp"):
        audio.with_name(audio.stem + suffix).write_text(SRT, encoding="utf-8")
    before = inventory(audio.parent)
    plan = build_execution_plan(root, mode=mode, skip_existing_subtitles=True)
    assert plan.executable
    store, outcomes, result, calls = execute_mixed(plan, tmp_path)
    skipped = next(t for t in plan.tracks if t.task.audio_path == audio)
    assert result["counts"]["skipped"] == 1
    assert result["counts"]["success" if mode == "full" else "transcribed"] == 1
    assert Counter(kind for kind, _ in calls)["asr"] == 1
    receipt = read_json(store.run_dir / "receipts" / (skipped.track_id + ".json"))
    assert receipt["provenance"] is None and "unverified_legacy" in receipt["warnings"]
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash)["complete"]
    assert inventory(audio.parent) == before


def test_legacy_diagnostics_aggregate_roles_and_reasons_with_safe_guidance(tmp_path):
    root = library(tmp_path)
    source = root / "RJ001/01_normal/Track01.srt"
    source.write_text(SRT, encoding="utf-8")
    invalid = root / "Other/Track99.wav"
    invalid.parent.mkdir()
    invalid.write_bytes(b"SYNTHETIC_UNRELATED_AUDIO")
    invalid.with_name("Track99-transcribe.srt").write_text(SRT, encoding="utf-8")
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    copy_issue = next(i for i in plan.issues if i.detail == "copy source lacks matching provenance")
    other_issue = next(i for i in plan.issues if i.detail == "no matching completed-stage provenance")
    rejection = PlanRejected((copy_issue,) * 269 + (other_issue,))
    intent = SimpleNamespace(catalog=plan, operation_id="synthetic", mode="full", root=root,
                             limit=None, selected=plan.selected_work_ids)
    report, summary = plan_rejection_diagnostic(intent, rejection)
    assert report["legacy_summary"] == [
        {"roles": ["final"], "reason": "unverified_copy_source", "issues": 269},
        {"roles": ["transcription"], "reason": "unverified_output", "issues": 1}]
    assert "final:unverified_copy_source=269" in summary
    assert "transcription:unverified_output=1" in summary
    assert "不會自動刪除" in summary and "取消問題作品" in summary
    assert len(summary) < 1500 and str(root) not in summary
    assert len(report["issues"]) == 270 and report["recovery_guidance"]
    assert SRT not in summary and SRT not in str(report)
