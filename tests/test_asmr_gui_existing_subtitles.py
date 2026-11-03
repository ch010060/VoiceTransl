"""GUI opt-in subtitle existence skips; no model or service inference is used."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import httpx
import pytest

from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan, make_provenance, path_key, _hash
from scripts.asmr_transcribe_lib.gui_plan_io import decode_plan, encode_plan, PlanTransportError
from scripts.asmr_transcribe_lib.gui_preflight import preflight
from scripts.asmr_transcribe_lib.gui_protocol import JSONLDecoder, ProtocolError
from scripts.asmr_transcribe_lib.gui_results import RunStore, read_json, reconcile_run

FORMATS = (".srt", ".ass", ".lrc", ".ssa", ".vtt", ".ASS", ".SRT")
SRT = "1\n00:00:00,000 --> 00:00:01,000\nSynthetic existing subtitle.\n"


def make_library(tmp_path, names=("RJ001/nested/voice01.wav",)):
    root = tmp_path / "library"
    root.mkdir()
    for name in names:
        audio = root / name
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"SYNTHETIC_AUDIO_NOT_INFERENCE")
    return root


def subtitle(audio, extension):
    path = audio.with_suffix(extension)
    content = {
        ".srt": SRT,
        ".ass": "[Script Info]\nTitle: Synthetic existing subtitle\n[Events]\nDialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Existing text\n",
        ".ssa": "[Script Info]\nTitle: Synthetic existing subtitle\n[Events]\nDialogue: Marked=0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Existing text\n",
        ".lrc": "[00:00.00]Synthetic existing subtitle\n",
        ".vtt": "WEBVTT\n\n00:00.000 --> 00:01.000\nSynthetic existing subtitle.\n",
    }[extension.lower()]
    path.write_text(content, encoding="utf-8")
    return path


def inventory(root):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def execute(plan, tmp_path):
    calls = []

    def transcriber(*, audio_path, transcribe_srt_path, **kwargs):
        calls.append(("asr", audio_path.name))
        transcribe_srt_path.write_text(SRT, encoding="utf-8")
        return transcribe_srt_path

    def translate(source, target):
        calls.append(("translation", Path(source).name))
        shutil.copyfile(source, target)
        return target

    def convert(source, target):
        calls.append(("opencc", Path(source).name))
        shutil.copyfile(source, target)
        return target

    store = RunStore(tmp_path / "run", str(uuid4()), plan, io.StringIO())
    store.start()
    outcomes = run_plan(plan, project_root=tmp_path, transcriber=transcriber,
        translator=translate if plan.mode == "full" else None,
        opencc_converter=convert, observer=store.observer)
    result = store.finish(outcomes)
    return store, outcomes, result, calls


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
@pytest.mark.parametrize("extension", FORMATS)
def test_optin_common_subtitles_are_read_only_skips(tmp_path, mode, extension):
    root = make_library(tmp_path)
    existing = subtitle(root / "RJ001/nested/voice01.wav", extension)
    before = inventory(root)
    plan = build_execution_plan(root, mode=mode, skip_existing_subtitles=True)
    assert plan.executable, plan.issues
    assert plan.settings["gui_existing_subtitles"] == "skip_common"
    track = plan.planned_tracks[0]
    assert track.task.action == "skip"
    assert track.task.existing_subtitle_path == existing
    assert decode_plan(encode_plan(plan)) == plan
    store, outcomes, result, calls = execute(plan, tmp_path)
    assert calls == []
    assert outcomes[0].status == "skipped" and outcomes[0].output_path == existing
    assert result["counts"]["skipped"] == 1 and result["counts"]["success"] == 0
    receipt = read_json(store.run_dir / "receipts" / (track.track_id + ".json"))
    assert receipt["output_type"] == "existing_subtitle"
    assert receipt["output_key"] == path_key(existing)
    assert receipt["output_hash"] == hashlib.sha256(existing.read_bytes()).hexdigest()
    assert receipt["provenance"] is None
    assert "unverified_legacy" in receipt["warnings"]
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash, 0)["counts"]["skipped"] == 1
    assert inventory(root) == before
    if extension.lower() != ".srt":
        assert not track.task.final_srt_path.exists()


@pytest.mark.parametrize("content", ["", "existing unparseable legacy captions"])
def test_optin_existing_srt_does_not_require_conversion_or_content_validation(tmp_path, content):
    root = make_library(tmp_path)
    existing = root / "RJ001/nested/voice01.srt"
    existing.write_text(content, encoding="utf-8")
    before = inventory(root)
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    store, _, result, calls = execute(plan, tmp_path)
    assert not calls and result["counts"]["skipped"] == 1
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash, 0)["complete"]
    assert inventory(root) == before


def test_unrelated_subtitle_filename_does_not_skip_audio(tmp_path):
    root = make_library(tmp_path)
    (root / "RJ001/nested/unrelated.ass").write_text("Unrelated subtitle", encoding="utf-8")
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    assert plan.executable and plan.tracks[0].task.action == "process"


@pytest.mark.parametrize("change", ["removed", "changed"])
def test_frozen_skip_rejects_changed_or_missing_subtitle(tmp_path, change):
    root = make_library(tmp_path)
    existing = subtitle(root / "RJ001/nested/voice01.wav", ".ass")
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    if change == "removed":
        existing.unlink()
    else:
        existing.write_text("Changed subtitle after frozen scan", encoding="utf-8")
    with pytest.raises(ValueError):
        plan.assert_ready()


@pytest.mark.parametrize("corruption", ["subtitle_bytes", "receipt_output_key", "receipt_output_hash"])
def test_skip_reconciliation_rejects_changed_bytes_or_receipt_identity(tmp_path, corruption):
    root = make_library(tmp_path)
    existing = subtitle(root / "RJ001/nested/voice01.wav", ".lrc")
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    store, _, _, _ = execute(plan, tmp_path)
    if corruption == "subtitle_bytes":
        existing.write_text("[00:00.00]Modified after receipt\n", encoding="utf-8")
    else:
        track_id = plan.tracks[0].track_id
        path = store.run_dir / "receipts" / (track_id + ".json")
        receipt = read_json(path)
        if corruption == "receipt_output_key":
            receipt["output_key"] = path_key(tmp_path / "foreign.lrc")
        else:
            receipt["output_hash"] = "0" * 64
        path.write_text(json.dumps(receipt), encoding="utf-8")
        # Recompute the outer receipt hash so verification must still enforce
        # frozen identity and actual subtitle bytes, rather than just this hash.
        digest = _hash(receipt)
        for filename in ("state.json", "result.json"):
            document = read_json(store.run_dir / filename)
            document["tracks"][track_id]["receipt_hash"] = digest
            (store.run_dir / filename).write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ProtocolError):
        reconcile_run(store.run_dir, store.run_id, plan.plan_hash, 0)


def test_frozen_skip_transport_rejects_foreign_subtitle_path(tmp_path):
    root = make_library(tmp_path)
    subtitle(root / "RJ001/nested/voice01.wav", ".ass")
    document = encode_plan(build_execution_plan(root, skip_existing_subtitles=True))
    document["plan"]["tracks"][0]["task"]["existing_subtitle_path"] = str(tmp_path / "foreign.ass")
    document["transport_hash"] = _hash({key: value for key, value in document.items() if key != "transport_hash"})
    with pytest.raises(PlanTransportError):
        decode_plan(document)


def test_existing_subtitle_policy_preserves_d02_groups_and_representative(tmp_path):
    root = make_library(tmp_path, ("RJ001/a/voice01.flac", "RJ001/b/voice01 SEなし.wav"))
    subtitle(root / "RJ001/a/voice01.flac", ".ass")
    legacy = build_execution_plan(root)
    opted = build_execution_plan(root, skip_existing_subtitles=True)
    assert opted.duplicate_groups == legacy.duplicate_groups
    assert [(t.track_id, t.representative_track_id) for t in opted.tracks] == [
        (t.track_id, t.representative_track_id) for t in legacy.tracks]
    assert opted.executable
    before = inventory(root)
    _, _, result, calls = execute(opted, tmp_path)
    assert calls == [] and result["counts"]["skipped"] == 2
    assert inventory(root) == before


def test_subset_frozen_transport_preserves_unselected_skip_catalog_tracks(tmp_path):
    root = make_library(tmp_path, ("WorkA/voice01.wav", "WorkB/nested/voice02.wav"))
    subtitle(root / "WorkA/voice01.wav", ".ass")
    subtitle(root / "WorkB/nested/voice02.wav", ".lrc")
    catalog = build_execution_plan(root, skip_existing_subtitles=True)
    work_a = next(work for work in catalog.works if work.relative_path == "WorkA")
    selected = build_execution_plan(root, selected_work_ids=[work_a.work_id], skip_existing_subtitles=True)
    assert selected.executable and len(selected.planned_tracks) == 1
    assert all(track.task.action == "skip" for track in selected.tracks)
    assert decode_plan(encode_plan(selected)) == selected
    assert selected.duplicate_groups == catalog.duplicate_groups
    before = inventory(root)
    _, _, result, calls = execute(selected, tmp_path)
    assert not calls and result["counts"]["skipped"] == 1
    assert inventory(root) == before


@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_d02_existing_srt_copy_still_requires_valid_source_provenance(tmp_path, mode):
    root = make_library(tmp_path, ("RJ001/a/voice01.flac", "RJ001/b/voice01 SEなし.wav"))
    existing = subtitle(root / "RJ001/a/voice01.flac", ".srt")
    unverified = build_execution_plan(root, mode=mode, skip_existing_subtitles=True)
    assert "UNVERIFIED_LEGACY" in {issue.code for issue in unverified.issues}
    owner = next(track for track in unverified.tracks if track.task.audio_path.suffix == ".flac")
    dependent = next(track for track in unverified.tracks if track.track_id != owner.track_id)
    assert owner.task.action == "skip"
    assert dependent.task.action == "copy_existing_subtitle"
    proof = make_provenance(unverified, owner.track_id, existing, "final")
    plan = build_execution_plan(root, mode=mode, skip_existing_subtitles=True, provenance=[proof])
    assert plan.executable, plan.issues
    assert plan.duplicate_groups == unverified.duplicate_groups
    assert decode_plan(encode_plan(plan)) == plan
    preserved = (existing.read_bytes(), existing.stat().st_mtime_ns)
    store, _, result, calls = execute(plan, tmp_path)
    assert calls == []
    assert result["counts"]["skipped"] == 1 and result["counts"]["copied"] == 1
    assert (existing.read_bytes(), existing.stat().st_mtime_ns) == preserved
    copied = plan.track_index[dependent.track_id]
    target = copied.task.final_srt_path if mode == "full" else copied.task.transcribe_srt_path
    assert target.read_bytes() == existing.read_bytes()
    assert reconcile_run(store.run_dir, store.run_id, plan.plan_hash, 0)["complete"]


def test_existing_subtitle_directory_is_rejected_without_inference(tmp_path):
    root = make_library(tmp_path)
    (root / "RJ001/nested/voice01.ass").mkdir()
    plan = build_execution_plan(root, skip_existing_subtitles=True)
    assert not plan.executable
    assert "INVALID_EXISTING_SUBTITLE" in {issue.code for issue in plan.issues}


def test_default_strict_ass_policy_is_preserved(tmp_path):
    root = make_library(tmp_path)
    subtitle(root / "RJ001/nested/voice01.wav", ".ass")
    plan = build_execution_plan(root)
    assert not plan.executable
    assert "UNSUPPORTED_SUBTITLE_CONVERSION" in {issue.code for issue in plan.issues}
    assert plan.tracks[0].task.action != "skip"


def test_all_skip_full_preflight_needs_no_sakura_or_asr_assets(tmp_path):
    root = make_library(tmp_path)
    subtitle(root / "RJ001/nested/voice01.wav", ".ass")
    plan = build_execution_plan(root, mode="full", skip_existing_subtitles=True)

    def forbidden_request(request):
        pytest.fail("an all-skip batch must not contact Sakura")

    def forbidden_asr_probe():
        pytest.fail("an all-skip batch must not probe or load ASR")

    with httpx.Client(transport=httpx.MockTransport(forbidden_request)) as client:
        result = preflight(plan, tmp_path / "missing-assets", client, runtime_probe=forbidden_asr_probe)
    assert result["available"]
    assert result["asr"]["required"] is False
    assert result["translation"]["required"] is False
    assert result["inference_proven"] is False


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    from scripts.asmr_gui.theme import apply_theme
    instance = QApplication.instance() or QApplication([])
    instance.setQuitOnLastWindowClosed(False)
    apply_theme(instance)
    return instance


@pytest.fixture(autouse=True)
def isolated_background_health(monkeypatch, request):
    """Execution assertions use synthetic workers and never contact local services."""
    if "app" not in request.fixturenames:
        return
    import scripts.asmr_gui.controller as controller_module
    from test_asmr_gui_v1 import SilentHealthMonitor
    monkeypatch.setattr(controller_module, "HealthMonitor", SilentHealthMonitor, raising=False)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_controller_owned_worker_skips_all_common_subtitles(app, tmp_path, mode):
    from PyQt6.QtTest import QSignalSpy
    from scripts.asmr_gui_runtime import ManagedWorker
    import test_asmr_gui_v1 as gui
    names = tuple(f"RJ001/nested{index}/voice{index:02d}.wav" for index in range(len(FORMATS)))
    root = make_library(tmp_path, names)
    subtitles = [subtitle(root / name, extension) for name, extension in zip(names, FORMATS)]
    before = inventory(root)
    workers = []

    def worker_factory(*args, **kwargs):
        worker = ManagedWorker(*args, **kwargs)
        workers.append(worker)
        return worker

    controller = gui.actual_controller(tmp_path, root, worker_factory=worker_factory)
    completed = QSignalSpy(controller.completed)
    try:
        assert controller.apply_settings({}, mode=mode)
        gui.scan_ready(app, controller, root)
        assert all(track.task.action == "skip" for track in controller.run_model.plan.planned_tracks)
        assert controller.start()
        gui.wait_for(app, lambda: len(completed) == 1 or controller.state == "ERROR", 25)
        assert controller.state == "FINISHED", controller.last_error
        assert completed[0][0]["counts"]["skipped"] == len(FORMATS)
        assert controller.run_model.summary()["success"] == 0
        assert controller.run_model.summary()["transcribed"] == 0
        assert controller.last_cleanup["cleanup_verified"]
        worker = workers[0]
        events = JSONLDecoder(worker.run_id).feed(worker.stdout_bytes)
        assert not any(event["stage"] in {"asr", "translation", "opencc"} for event in events)
        for track, path in zip(worker.plan.planned_tracks, subtitles):
            receipt = read_json(worker.run_dir / "receipts" / (track.track_id + ".json"))
            assert receipt["output_type"] == "existing_subtitle" and receipt["output_key"] == path_key(path)
            assert receipt["provenance"] is None and "unverified_legacy" in receipt["warnings"]
            assert "unverified_legacy" in controller.run_model.tracks[track.track_id].warnings
        assert inventory(root) == before
    finally:
        gui.finish_controller(app, controller)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows Job containment")
@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_controller_mixed_work_executes_fresh_track_only(app, tmp_path, mode):
    from PyQt6.QtTest import QSignalSpy
    from scripts.asmr_gui_runtime import ManagedWorker
    import test_asmr_gui_v1 as gui
    root = make_library(tmp_path, ("RJ001/voice01.wav", "RJ001/nested/voice02.wav"))
    existing = subtitle(root / "RJ001/voice01.wav", ".ass")
    preserved = (existing.read_bytes(), existing.stat().st_mtime_ns)
    workers = []

    def worker_factory(*args, **kwargs):
        worker = ManagedWorker(*args, **kwargs)
        workers.append(worker)
        return worker

    controller = gui.actual_controller(tmp_path, root, worker_factory=worker_factory)
    completed = QSignalSpy(controller.completed)
    try:
        assert controller.apply_settings({}, mode=mode)
        gui.scan_ready(app, controller, root)
        assert controller.start()
        gui.wait_for(app, lambda: len(completed) == 1 or controller.state == "ERROR", 25)
        assert controller.state == "FINISHED", controller.last_error
        result = completed[0][0]
        assert result["counts"]["skipped"] == 1
        assert result["counts"]["success" if mode == "full" else "transcribed"] == 1
        worker = workers[0]
        skipped = next(t for t in worker.plan.tracks if t.task.audio_path.name == "voice01.wav")
        fresh = next(t for t in worker.plan.tracks if t.task.audio_path.name == "voice02.wav")
        events = JSONLDecoder(worker.run_id).feed(worker.stdout_bytes)
        asr_ids = {event["track_id"] for event in events if event["stage"] == "asr"}
        assert asr_ids == {fresh.track_id}
        assert not any(event["track_id"] == skipped.track_id and event["stage"] in {"asr", "translation", "copy"} for event in events)
        assert (existing.read_bytes(), existing.stat().st_mtime_ns) == preserved
        assert not skipped.task.final_srt_path.exists()
        assert not skipped.task.transcribe_srt_path.exists()
        assert controller.last_cleanup["cleanup_verified"]
    finally:
        gui.finish_controller(app, controller)
