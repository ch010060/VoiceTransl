from dataclasses import FrozenInstanceError
from pathlib import Path
import threading
import os
import time

import pytest

from scripts.asmr_transcribe_lib.gui_plan import (
    PlanRejected, build_execution_plan, make_provenance,
)
from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.runtime import CancellationToken
from scripts.asmr_transcribe_lib.srt_utils import parse_srt_file


SRT = "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n"


def audio(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic audio " + relative.encode())
    return path


def execute(plan, *, fail=None, token=None, observer=None, parallel=None):
    calls = []

    def asr(*, audio_path, transcribe_srt_path, **kwargs):
        calls.append(audio_path)
        if fail and fail(audio_path):
            raise RuntimeError("synthetic ASR failure")
        callback = kwargs.get("transcription_progress_callback")
        if callback:
            callback(1.0, 1.0, 1.0)
        transcribe_srt_path.write_text(SRT, encoding="utf-8")
        return transcribe_srt_path

    def copy(source, target):
        target.write_bytes(source.read_bytes())
        return target

    results = run_plan(
        plan, project_root=plan.root, transcriber=asr,
        translator=copy if plan.mode == "full" else None,
        opencc_converter=copy, parallel=parallel,
        observer=observer, cancel_token=token,
    )
    return results, calls


@pytest.mark.parametrize("relative", ["work/a-transcribe.wav", "work/a-translate.wav", "work/a-opencc.wav"])
@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_f01_namespace_rejected_before_any_work(tmp_path, relative, mode):
    audio(tmp_path, "work/a.wav")
    audio(tmp_path, relative)
    plan = build_execution_plan(tmp_path, mode=mode)
    assert "OUTPUT_NAMESPACE_COLLISION" in {i.code for i in plan.issues}
    with pytest.raises(PlanRejected):
        execute(plan)
    assert not list(tmp_path.rglob("*.srt"))


def test_f02_root_same_stem_collision(tmp_path):
    audio(tmp_path, "same.wav")
    audio(tmp_path, "same.flac")
    assert not build_execution_plan(tmp_path).executable


@pytest.mark.parametrize("extension", ["vtt", "ass", "ssa", "lrc"])
def test_f03_alt_subtitle_is_blocked_and_preserved(tmp_path, extension):
    source = audio(tmp_path, "work/a.wav")
    alternate = source.with_suffix("." + extension)
    alternate.write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nhello", encoding="utf-8")
    before = alternate.read_bytes()
    plan = build_execution_plan(tmp_path)
    assert "UNSUPPORTED_SUBTITLE_CONVERSION" in {i.code for i in plan.issues}
    with pytest.raises(PlanRejected):
        execute(plan)
    assert alternate.read_bytes() == before
    assert not source.with_suffix(".srt").exists()


@pytest.mark.parametrize("mode, suffix", [("full", ".srt"), ("transcribe_only", "-transcribe.srt")])
def test_d02_r1_r5_r6_duplicate_waits_for_mode_specific_output(tmp_path, mode, suffix):
    a = audio(tmp_path, "work/mp3/Track01.mp3")
    b = audio(tmp_path, "work/wav/Track01.wav")
    originals = {p: p.read_bytes() for p in [a, b]}
    plan = build_execution_plan(tmp_path, mode=mode)
    results, calls = execute(plan)
    assert calls == [b]
    assert sorted(r.status for r in results) == ["copied", "success" if mode == "full" else "transcribed"]
    for p in [a, b]:
        target = p.with_suffix(".srt") if mode == "full" else p.with_name(p.stem + suffix)
        assert parse_srt_file(target)
        assert p.read_bytes() == originals[p]
    assert plan.tracks[0].dependencies == (plan.tracks[1].track_id,)


def test_f05_limit_excluding_representative_is_rejected(tmp_path):
    audio(tmp_path, "work/a/Track01.mp3")
    audio(tmp_path, "work/b/Other02.wav")
    audio(tmp_path, "work/c/Track01.flac")
    plan = build_execution_plan(tmp_path, limit=1)
    assert "DEPENDENCY_EXCLUDED" in {i.code for i in plan.issues}
    with pytest.raises(PlanRejected):
        execute(plan)


def test_f06_independent_failure_isolated_and_every_track_terminal(tmp_path):
    audio(tmp_path, "work/a.wav")
    audio(tmp_path, "work/b.wav")
    events = []
    results, _ = execute(build_execution_plan(tmp_path), fail=lambda p: p.stem == "a", observer=events.append)
    assert [r.status for r in results] == ["failed", "success"]
    assert (tmp_path / "work/b.srt").exists()
    assert len([e for e in events if e.event in {"track_finished", "track_failed"}]) == 2


@pytest.mark.parametrize("formats, expected", [(["mp3", "wav", "flac"], "flac"), (["mp3", "wav"], "wav"), (["mp3"], "mp3")])
def test_d02_r2_format_priority(tmp_path, formats, expected):
    for fmt in formats:
        audio(tmp_path, f"work/{fmt}/Track01.{fmt}")
    _, calls = execute(build_execution_plan(tmp_path))
    assert len(calls) == 1 and calls[0].suffix == "." + expected


def test_d02_r3_se_nashi_and_fuzzy_grouping_preserved(tmp_path):
    audio(tmp_path, "work/a/【Track01】お耳を優しくなでなでします.mp3")
    audio(tmp_path, "work/b/【Track01】お耳を優しくなでなでします【SEなし】.flac")
    _, calls = execute(build_execution_plan(tmp_path))
    assert len(calls) == 1 and calls[0].suffix == ".flac"


def test_d02_r4_different_works_remain_independent(tmp_path):
    audio(tmp_path, "work1/Track01.wav")
    audio(tmp_path, "work2/Track01.flac")
    _, calls = execute(build_execution_plan(tmp_path))
    assert len(calls) == 2


def test_d02_r7_collision_preserves_matcher_group_and_outputs(tmp_path):
    a = audio(tmp_path, "work/wav/Track01.wav")
    audio(tmp_path, "work/flac/Track01.flac")
    audio(tmp_path, "work/flac/Track01-transcribe.wav")
    existing = tmp_path / "work/wav/Track01-translate.srt"
    existing.write_text(SRT, encoding="utf-8")
    plan = build_execution_plan(tmp_path)
    assert any(len(g.members) == 2 for g in plan.duplicate_groups)
    assert any(i.code == "OUTPUT_NAMESPACE_COLLISION" for i in plan.issues)
    with pytest.raises(PlanRejected):
        execute(plan)
    assert existing.read_text(encoding="utf-8") == SRT


def test_d02_same_directory_formats_share_one_owned_output(tmp_path):
    audio(tmp_path, "work/Track01.wav")
    audio(tmp_path, "work/Track01.mp3")
    plan = build_execution_plan(tmp_path)
    assert plan.executable
    results, calls = execute(plan)
    assert len(calls) == 1 and calls[0].suffix == ".wav"
    assert sorted(r.status for r in results) == ["copied", "success"]


def test_d02_r8_failed_representative_never_copied(tmp_path):
    audio(tmp_path, "work/mp3/Track01.mp3")
    audio(tmp_path, "work/wav/Track01.wav")
    results, _ = execute(build_execution_plan(tmp_path), fail=lambda p: True)
    assert [r.status for r in results] == ["not_started", "failed"]
    assert results[0].reason == "dependency_failed"
    assert not list(tmp_path.rglob("*.srt"))


def test_identity_empty_nested_and_real_root_group(tmp_path):
    audio(tmp_path, "_root/nested/a.wav")
    audio(tmp_path, "a.wav")
    (tmp_path / "empty").mkdir()
    a, b = build_execution_plan(tmp_path), build_execution_plan(tmp_path)
    assert a.plan_hash == b.plan_hash and a.plan_id == b.plan_id
    assert a.works == b.works and a.tracks == b.tracks
    assert len(a.works) == 3
    assert len({w.work_id for w in a.works}) == 3
    assert {w.kind for w in a.works} == {"directory", "synthetic_root"}
    with pytest.raises(FrozenInstanceError):
        a.mode = "other"


def test_only_work_selection_and_stable_track_ids(tmp_path):
    audio(tmp_path, "one/a.wav")
    audio(tmp_path, "two/b.wav")
    all_plan = build_execution_plan(tmp_path)
    selected = build_execution_plan(tmp_path, selected_work_ids=(all_plan.works[1].work_id,))
    assert selected.tracks == all_plan.tracks
    assert [t.track_id for t in selected.planned_tracks] == [all_plan.tracks[1].track_id]


def test_legacy_transcript_blocked_without_touching_it(tmp_path):
    p = audio(tmp_path, "work/a.wav")
    cache = p.with_name("a-transcribe.srt")
    cache.write_text(SRT, encoding="utf-8")
    plan = build_execution_plan(tmp_path)
    assert any(i.code == "UNVERIFIED_LEGACY" for i in plan.issues)
    with pytest.raises(PlanRejected):
        execute(plan)
    assert cache.read_text(encoding="utf-8") == SRT


def test_matching_provenance_resume_observed_without_asr(tmp_path):
    p = audio(tmp_path, "work/a.wav")
    cache = p.with_name("a-transcribe.srt")
    cache.write_text(SRT, encoding="utf-8")
    initial = build_execution_plan(tmp_path)
    receipt = make_provenance(initial, initial.tracks[0].track_id, cache, "transcription")
    plan = build_execution_plan(tmp_path, provenance=(receipt,))
    events = []
    results, calls = execute(plan, observer=events.append)
    assert calls == [] and results[0].status == "success"
    assert "asr_reused" in [e.event for e in events]
    assert not any(e.stage == "asr" for e in events)


def test_cancellation_before_start_and_before_translation(tmp_path):
    audio(tmp_path, "work/a.wav")
    token = CancellationToken()
    token.cancel()
    results, calls = execute(build_execution_plan(tmp_path), token=token)
    assert calls == [] and results[0].status == "not_started"
    token = CancellationToken()

    def observe(event):
        if event.event == "asr_completed":
            token.cancel()

    results, _ = execute(build_execution_plan(tmp_path), token=token, observer=observe)
    assert results[0].status == "cancelled"
    assert not (tmp_path / "work/a.srt").exists()


def test_asr_100_never_synthesizes_translation_progress(tmp_path):
    audio(tmp_path, "work/a.wav")
    events = []
    results, _ = execute(build_execution_plan(tmp_path), observer=events.append)
    stages = [e.stage for e in events if e.event == "stage_changed"]
    assert all(stage in stages for stage in ["waiting_asr", "asr", "translation", "opencc", "finalize"])
    assert results[0].status == "success"
    assert not any(e.progress is not None and e.stage == "translation" for e in events)


@pytest.mark.skipif(os.name != "nt", reason="Windows path alias behavior")
def test_windows_case_insensitive_namespace_aliases(tmp_path):
    audio(tmp_path, "work/A.wav")
    audio(tmp_path, "work/a-transcribe.wav")
    assert any(i.code == "OUTPUT_NAMESPACE_COLLISION" for i in build_execution_plan(tmp_path).issues)


def fake_asr(*, audio_path, transcribe_srt_path, **kwargs):
    transcribe_srt_path.write_text(SRT, encoding="utf-8")
    return transcribe_srt_path


def fake_copy(source, target):
    target.write_bytes(source.read_bytes())
    return target


@pytest.mark.parametrize("stage", ["translation", "finalize"])
def test_cancel_from_stage_observer_prevents_side_effect(tmp_path, stage):
    audio(tmp_path, "work/a.wav")
    token = CancellationToken()
    def observe(event):
        if event.event == "stage_changed" and event.stage == stage:
            token.cancel()
    results, _ = execute(build_execution_plan(tmp_path), token=token, observer=observe)
    assert results[0].status == "cancelled"
    assert not (tmp_path / "work/a.srt").exists()
    if stage == "translation":
        assert not (tmp_path / "work/a-translate.srt").exists()


def test_asr100_then_translation_failure_is_failed(tmp_path):
    audio(tmp_path, "work/a.wav")
    events = []
    def transcribe(**kwargs):
        kwargs["transcription_progress_callback"](1.0, 10.0, 10.0)
        return fake_asr(**kwargs)
    def fail(source, target):
        raise RuntimeError("synthetic translation failure")
    results = run_plan(build_execution_plan(tmp_path), project_root=tmp_path,
                       transcriber=transcribe, translator=fail, opencc_converter=fake_copy,
                       observer=events.append)
    assert results[0].status == "failed"
    assert any(e.event == "asr_progress" and e.progress[0] == 1 for e in events)
    assert not any(e.status == "success" for e in events)
    assert not (tmp_path / "work/a.srt").exists()


def test_read_only_source_hardlinks_across_works_allowed(tmp_path):
    first = audio(tmp_path, "one/a.wav")
    second = tmp_path / "two/a.wav"
    second.parent.mkdir()
    os.link(first, second)
    plan = build_execution_plan(tmp_path)
    assert plan.executable
    results, calls = execute(plan)
    assert len(calls) == 2 and all(r.status == "success" for r in results)


def test_output_hardlink_alias_to_source_is_rejected(tmp_path):
    first = audio(tmp_path, "work/a.wav")
    os.link(first, first.with_name("a-transcribe.srt"))
    assert any(i.code == "OUTPUT_NAMESPACE_COLLISION" for i in build_execution_plan(tmp_path).issues)


def test_dependent_legacy_transcription_is_blocked_before_asr(tmp_path):
    dependent = audio(tmp_path, "work/mp3/Track01.mp3")
    audio(tmp_path, "work/wav/Track01.wav")
    target = dependent.with_name("Track01-transcribe.srt")
    target.write_text(SRT, encoding="utf-8")
    plan = build_execution_plan(tmp_path, mode="transcribe_only")
    assert any(i.code == "UNVERIFIED_LEGACY" and target in i.paths for i in plan.issues)
    with pytest.raises(PlanRejected):
        execute(plan)
    assert target.read_text(encoding="utf-8") == SRT


def test_skipped_representative_without_transcription_is_plan_error(tmp_path):
    audio(tmp_path, "work/mp3/Track01.mp3")
    rep = audio(tmp_path, "work/wav/Track01.wav")
    rep.with_suffix(".srt").write_text(SRT, encoding="utf-8")
    plan = build_execution_plan(tmp_path, mode="transcribe_only")
    assert any(i.code == "DEPENDENCY_OUTPUT_UNAVAILABLE" for i in plan.issues)


@pytest.mark.parametrize("settings", [
    {"parallel": 0}, {"transcribe_parallel": -1}, {"transcribe_batch_size": 0},
    {"translate_chunk_size": 0}, {"transcribe_backend": "unknown"},
    {"fast_transcribe": "true"}, {"parallel": "2"}, {"parallel": True},
])
def test_invalid_settings_rejected_before_execution(tmp_path, settings):
    audio(tmp_path, "work/a.wav")
    with pytest.raises(ValueError):
        build_execution_plan(tmp_path, settings=settings)


def test_frozen_plan_refuses_source_or_output_changes(tmp_path):
    a = audio(tmp_path, "work/a.wav")
    plan = build_execution_plan(tmp_path)
    a.write_bytes(b"changed")
    with pytest.raises(PlanRejected, match="PLAN_STALE"):
        execute(plan)
    plan = build_execution_plan(tmp_path)
    a.with_suffix(".srt").write_text(SRT, encoding="utf-8")
    with pytest.raises(PlanRejected, match="PLAN_STALE"):
        execute(plan)


@pytest.mark.parametrize("change", ["source", "settings", "output"])
def test_provenance_staleness_does_not_reuse_legacy(tmp_path, change):
    a = audio(tmp_path, "work/a.wav")
    cache = a.with_name("a-transcribe.srt")
    cache.write_text(SRT, encoding="utf-8")
    initial = build_execution_plan(tmp_path)
    proof = make_provenance(initial, initial.tracks[0].track_id, cache, "transcription")
    settings = None
    if change == "source":
        a.write_bytes(b"changed source")
    elif change == "settings":
        settings = {"transcribe_batch_size": 8}
    else:
        cache.write_text(SRT.replace("hello", "changed"), encoding="utf-8")
    plan = build_execution_plan(tmp_path, settings=settings, provenance=(proof,))
    assert any(i.code == "UNVERIFIED_LEGACY" for i in plan.issues)


def test_d02_tie_break_and_same_directory_copy_in_transcribe_only(tmp_path):
    a = audio(tmp_path, "work/a/Track01.flac")
    audio(tmp_path, "work/z/Track01.flac")
    results, calls = execute(build_execution_plan(tmp_path, mode="transcribe_only"))
    assert calls == [a] and sorted(r.status for r in results) == ["copied", "transcribed"]


@pytest.mark.parametrize("concurrency", [1, 2])
def test_safe_execution_obeys_asr_concurrency(tmp_path, concurrency):
    audio(tmp_path, "work/a.wav")
    audio(tmp_path, "work/b.wav")
    plan = build_execution_plan(tmp_path, settings={"parallel": 2, "transcribe_parallel": concurrency})
    lock = threading.Lock()
    active = maximum = 0
    barrier = threading.Barrier(2)
    def asr(**kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        if concurrency == 2:
            barrier.wait(timeout=2)
        else:
            time.sleep(0.02)
        result = fake_asr(**kwargs)
        with lock:
            active -= 1
        return result
    results = run_plan(plan, project_root=tmp_path, transcriber=asr,
                       translator=fake_copy, opencc_converter=fake_copy)
    assert all(r.status == "success" for r in results)
    assert maximum == concurrency


def test_safe_asr_translation_overlap(tmp_path):
    audio(tmp_path, "work/a.wav")
    audio(tmp_path, "work/b.wav")
    plan = build_execution_plan(tmp_path, settings={"parallel": 2})
    tl_started, overlap = threading.Event(), threading.Event()
    first_path = []
    lock = threading.Lock()
    def asr(**kwargs):
        with lock:
            first = not first_path
            if first:
                first_path.append(kwargs["audio_path"])
        if not first:
            assert tl_started.wait(2)
            overlap.set()
        return fake_asr(**kwargs)
    def tl(source, target):
        if source == first_path[0].with_name(first_path[0].stem + "-transcribe.srt"):
            tl_started.set()
            assert overlap.wait(2)
        return fake_copy(source, target)
    results = run_plan(plan, project_root=tmp_path, transcriber=asr,
                       translator=tl, opencc_converter=fake_copy)
    assert overlap.is_set() and all(r.status == "success" for r in results)


@pytest.mark.parametrize("parallel", [1, 3])
def test_all_skip_copy_resume_paths_observed_in_both_schedulers(tmp_path, parallel):
    skipped = audio(tmp_path, "work/skip.wav")
    skipped.with_suffix(".srt").write_text(SRT, encoding="utf-8")
    audio(tmp_path, "work/a/Track01.mp3")
    audio(tmp_path, "work/b/Track01.flac")
    events = []
    plan = build_execution_plan(tmp_path, settings={"parallel": parallel})
    results, _ = execute(plan, observer=events.append)
    assert sorted(r.status for r in results) == ["copied", "skipped", "success"]
    terminal = [e for e in events if e.event == "track_finished"]
    assert len(terminal) == 3 and len({e.track_id for e in terminal}) == 3
    assert next(r for r in results if r.status == "skipped").warnings == ("unverified_legacy",)


def test_observer_failure_cancels_shared_token(tmp_path):
    from scripts.asmr_transcribe_lib.runtime import ObserverError
    audio(tmp_path, "work/a.wav")
    token = CancellationToken()
    def broken(event):
        raise RuntimeError("observer broken")
    with pytest.raises(ObserverError):
        execute(build_execution_plan(tmp_path), token=token, observer=broken)
    assert token.cancelled


def test_adapter_unowned_path_is_rejected_before_correction(tmp_path):
    audio(tmp_path, "work/a.wav")
    unrelated = tmp_path / "completed.srt"
    unrelated.write_text(SRT, encoding="utf-8")
    def asr(**kwargs):
        return unrelated
    results = run_plan(build_execution_plan(tmp_path), project_root=tmp_path,
                       transcriber=asr, translator=fake_copy, opencc_converter=fake_copy)
    assert results[0].status == "failed"
    assert unrelated.read_text(encoding="utf-8") == SRT
    assert not (tmp_path / "work/a.srt").exists()


def test_final_appearing_at_finalize_is_never_overwritten(tmp_path):
    audio(tmp_path, "work/a.wav")
    final = tmp_path / "work/a.srt"
    def observe(event):
        if event.stage == "finalize":
            final.write_text("independent existing subtitle", encoding="utf-8")
    results, _ = execute(build_execution_plan(tmp_path), observer=observe)
    assert results[0].status == "failed"
    assert final.read_text(encoding="utf-8") == "independent existing subtitle"


def test_transcript_appearing_during_start_is_preserved(tmp_path):
    audio(tmp_path, "work/a.wav")
    target = tmp_path / "work/a-transcribe.srt"
    def observe(event):
        if event.event == "track_started":
            target.write_text(SRT, encoding="utf-8")
    results, calls = execute(build_execution_plan(tmp_path), observer=observe)
    assert calls == [] and results[0].status == "failed"
    assert target.read_text(encoding="utf-8") == SRT


def test_limit_preserves_explicit_excluded_tracks_and_later_skips(tmp_path):
    audio(tmp_path, "work/a.wav")
    audio(tmp_path, "work/b.wav")
    skipped = audio(tmp_path, "work/z.wav")
    skipped.with_suffix(".srt").write_text(SRT, encoding="utf-8")
    plan = build_execution_plan(tmp_path, limit=1)
    assert len(plan.tracks) == 3 and len(plan.excluded_track_ids) == 1
    results, calls = execute(plan)
    assert len(calls) == 1 and [r.status for r in results] == ["success", "skipped"]


def test_cancel_while_waiting_for_asr_gate_does_not_start_second_asr(tmp_path):
    audio(tmp_path, "work/a.wav")
    audio(tmp_path, "work/b.wav")
    token = CancellationToken()
    second_waiting = threading.Event()
    calls = []
    plan = build_execution_plan(tmp_path, settings={"parallel": 2})
    waiting = set()
    def observe(event):
        if event.stage == "waiting_asr":
            waiting.add(event.track_id)
            if len(waiting) == 2:
                second_waiting.set()
    def asr(**kwargs):
        calls.append(kwargs["audio_path"])
        assert second_waiting.wait(2)
        token.cancel()
        return fake_asr(**kwargs)
    results = run_plan(plan, project_root=tmp_path, transcriber=asr,
                       translator=fake_copy, opencc_converter=fake_copy,
                       observer=observe, cancel_token=token)
    assert len(calls) == 1 and all(r.status == "cancelled" for r in results)


def test_unverified_existing_final_is_skip_not_verified_completion(tmp_path):
    a = audio(tmp_path, "work/a.wav")
    a.with_suffix(".srt").write_bytes(b"invalid old subtitle")
    plan = build_execution_plan(tmp_path)
    results, calls = execute(plan)
    assert calls == [] and results[0].status == "skipped"
    assert results[0].warnings == ("unverified_legacy",)
    assert a.with_suffix(".srt").read_bytes() == b"invalid old subtitle"


def test_invalid_representative_output_does_not_allow_copy(tmp_path):
    audio(tmp_path, "work/mp3/Track01.mp3")
    audio(tmp_path, "work/wav/Track01.wav")
    def asr(**kwargs):
        kwargs["transcribe_srt_path"].write_text("WEBVTT", encoding="utf-8")
        return kwargs["transcribe_srt_path"]
    plan = build_execution_plan(tmp_path)
    results = run_plan(plan, project_root=tmp_path, transcriber=asr,
                       translator=fake_copy, opencc_converter=fake_copy)
    assert [r.status for r in results] == ["not_started", "failed"]
    assert not (tmp_path / "work/mp3/Track01.srt").exists()


def test_legacy_run_tasks_still_raises_instead_of_changing_cli_exit_contract(tmp_path):
    from scripts.asmr_transcribe_lib.pipeline import run_tasks
    from scripts.asmr_transcribe_lib.planner import discover_audio_tasks
    from scripts.asmr_transcribe_lib.local_faster_whisper import FasterWhisperConfig
    audio(tmp_path, "work/a.wav")
    def fail(**kwargs):
        raise RuntimeError("legacy failure")
    with pytest.raises(RuntimeError, match="legacy failure"):
        run_tasks(discover_audio_tasks(tmp_path), project_root=tmp_path,
                  faster_whisper_config=FasterWhisperConfig(model="fake", language="ja"),
                  transcriber=fail, translator=fake_copy, parallel=2)


def test_plan_effective_defaults_match_cli_without_mutable_settings(tmp_path):
    audio(tmp_path, "work/a.wav")
    plan = build_execution_plan(tmp_path)
    assert {key: plan.settings[key] for key in ["parallel", "transcribe_parallel", "transcribe_batch_size", "translate_chunk_size"]} == {
        "parallel": 5, "transcribe_parallel": 1, "transcribe_batch_size": 12, "translate_chunk_size": 20,
    }
    edited = plan.settings
    edited["parallel"] = 99
    assert plan.settings["parallel"] == 5
    with pytest.raises(TypeError):
        plan.track_index["another"] = plan.tracks[0]


def test_existing_read_only_subtitle_hardlinks_do_not_create_writers(tmp_path):
    a = audio(tmp_path, "one/a.wav")
    b = audio(tmp_path, "two/b.wav")
    a.with_suffix(".srt").write_text(SRT, encoding="utf-8")
    os.link(a.with_suffix(".srt"), b.with_suffix(".srt"))
    plan = build_execution_plan(tmp_path)
    results, calls = execute(plan)
    assert calls == [] and all(r.status == "skipped" for r in results)
    assert all(use.access == "read" for use in plan.output_ownership)


def test_empty_work_preserved_but_empty_execution_rejected(tmp_path):
    (tmp_path / "empty").mkdir()
    plan = build_execution_plan(tmp_path)
    assert len(plan.works) == 1 and plan.works[0].track_ids == ()
    assert not plan.executable
    with pytest.raises(PlanRejected, match="EMPTY_EXECUTION_PLAN"):
        execute(plan)


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_collision_rejection_calls_neither_asr_nor_copy(tmp_path, mode):
    audio(tmp_path, "work/a.wav")
    audio(tmp_path, "work/a-transcribe.wav")
    calls = []
    def unexpected(*args, **kwargs):
        calls.append(1)
        raise AssertionError("planning must fail before any work")
    with pytest.raises(PlanRejected, match="OUTPUT_NAMESPACE_COLLISION"):
        run_plan(build_execution_plan(tmp_path, mode=mode), project_root=tmp_path,
                 transcriber=unexpected, translator=unexpected if mode == "full" else None, opencc_converter=unexpected)
    assert calls == []


def test_effective_model_prompt_are_frozen_and_kotoba_config_is_mode_specific(tmp_path):
    from scripts.asmr_transcribe_lib.gui_execution import _config
    from scripts.asmr_transcribe_lib.cli import DEFAULT_ASMR_INITIAL_PROMPT
    from scripts.asmr_transcribe_lib.kotoba_faster_whisper import KotobaFasterWhisperConfig
    audio(tmp_path, "work/a.wav")
    plan = build_execution_plan(tmp_path)
    assert plan.settings["transcribe_model"] == "faster-whisper-large-v3"
    assert plan.settings["initial_prompt"] == DEFAULT_ASMR_INITIAL_PROMPT
    assert plan.settings["xxl_chunk_length"] == 20
    alternate = build_execution_plan(tmp_path, settings={"transcribe_backend": "kotoba"})
    config = _config(alternate, tmp_path)
    assert isinstance(config, KotobaFasterWhisperConfig)
    assert config.chunk_length == 15 and config.device == "cuda"


def test_d02_nonidentical_filenames_still_use_fuzzy_matching(tmp_path):
    stem = "01_プロローグ「眠る前の優しいお話と耳元での静かなささやき」"
    representative = audio(tmp_path, "work/01/" + stem + ".flac")
    audio(tmp_path, "work/02/【BGMなし】" + stem + ".wav")
    results, calls = execute(build_execution_plan(tmp_path))
    assert calls == [representative]
    assert sorted(r.status for r in results) == ["copied", "success"]


def test_d02_transitive_grouping_preserved(tmp_path):
    audio(tmp_path, "work/a/abcdefghijklmnopqrstuvwx.wav")
    audio(tmp_path, "work/b/XXcdefghijklmnopqrstuvwx.wav")
    audio(tmp_path, "work/c/XXXXefghijklmnopqrstuvwx.wav")
    plan = build_execution_plan(tmp_path)
    assert len(plan.duplicate_groups) == 1 and len(plan.duplicate_groups[0].members) == 3
    results, calls = execute(plan)
    assert len(calls) == 1 and sorted(r.status for r in results) == ["copied", "copied", "success"]


def test_d02_numeric_guards_still_prevent_cross_track_merge(tmp_path):
    audio(tmp_path, "work/a/Track01_眠る前の静かなささやきと優しいお話.wav")
    audio(tmp_path, "work/b/Track02_眠る前の静かなささやきと優しいお話.wav")
    results, calls = execute(build_execution_plan(tmp_path))
    assert len(calls) == 2 and all(r.status == "success" for r in results)
