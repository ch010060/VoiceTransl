"""Frozen handoff, read-only availability and real isolated-worker integration."""
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import httpx
import pytest

from scripts.asmr_gui_runtime import ManagedWorker, WindowsJob
from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan
from scripts.asmr_transcribe_lib.gui_plan_io import (
    PlanTransportError, decode_plan, encode_plan, load_plan, save_plan,
)
from scripts.asmr_transcribe_lib.gui_preflight import preflight
from scripts.asmr_transcribe_lib.gui_protocol import JSONLDecoder
from scripts.asmr_transcribe_lib.gui_results import RunStore, read_json
from scripts.asmr_transcribe_lib.gui_execution import run_plan
from scripts.asmr_transcribe_lib.gui_native_adapter import strict_xxl_transcriber
from scripts.asmr_transcribe_lib.gui_ownership import NamespaceLease

PROJECT = Path(__file__).resolve().parents[1]
DRIVER = PROJECT / "tests" / "fixtures" / "asmr_gui_worker_driver.py"
WORKER = PROJECT / "scripts" / "asmr_gui_worker.py"
RUNTIME = PROJECT / "scripts" / "asmr_gui_runtime.py"


def plan_fixture(tmp_path, mode="full", names=("voice01.flac",), **settings):
    root = tmp_path / "library"
    work = root / "RJ000001"
    work.mkdir(parents=True)
    for name in names:
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-audio-not-inference")
    return build_execution_plan(root, mode=mode, settings=settings)


def local_assets(tmp_path, model="large-v3"):
    assets = tmp_path / "assets" / "whisper-faster"
    assets.mkdir(parents=True)
    (assets / "whisper-faster.exe").write_bytes(b"fake-executable-never-run")
    directory = assets / ("faster-whisper-" + model)
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    (directory / "model.bin").write_bytes(b"synthetic-model-never-loaded")
    for name in ("tokenizer.json", "vocabulary.json", "preprocessor_config.json"):
        (directory / name).write_text("{}")
    return assets.parent


def test_frozen_plan_roundtrip_d02_without_discovery(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path, names=("a/voice01.flac", "b/voice01 SEなし.wav", "c/voice01.mp3"))
    document = encode_plan(plan)
    monkeypatch.setattr("scripts.asmr_transcribe_lib.gui_plan.discover_audio_tasks", lambda *a: pytest.fail("must not rescan"))
    decoded = decode_plan(document, document["transport_hash"])
    assert decoded == plan
    assert decoded.duplicate_groups == plan.duplicate_groups
    assert decoded.output_ownership == plan.output_ownership


@pytest.mark.parametrize("change", ["digest", "major", "graph", "namespace", "settings"])
def test_frozen_plan_rejects_changed_transport(tmp_path, change):
    plan = plan_fixture(tmp_path)
    document = encode_plan(plan)
    expected = document["transport_hash"]
    if change == "digest":
        expected = "0" * 64
    elif change == "major":
        document["schema_version"] = "2.0"
    elif change == "graph":
        document["plan"]["tracks"][0]["dependencies"] = [document["plan"]["tracks"][0]["track_id"]]
    elif change == "namespace":
        document["plan"]["tracks"][0]["task"]["final_srt_path"] = str(tmp_path / "foreign.srt")
    else:
        document["plan"]["settings_json"] = "{}"
    with pytest.raises(PlanTransportError):
        decode_plan(document, expected)


@pytest.mark.parametrize("change", ["new_audio", "deleted_audio", "changed_audio", "new_output"])
def test_frozen_plan_stale_before_inference(tmp_path, change):
    plan = plan_fixture(tmp_path)
    path = tmp_path / "state" / "plan.json"
    digest = save_plan(plan, path)
    task = plan.tracks[0].task
    if change == "new_audio":
        task.audio_path.with_name("new02.wav").write_bytes(b"new")
    elif change == "deleted_audio":
        task.audio_path.unlink()
    elif change == "changed_audio":
        task.audio_path.write_bytes(b"changed")
    else:
        task.final_srt_path.write_text("foreign", encoding="utf-8")
    with pytest.raises(ValueError):
        loaded = load_plan(path, digest)
        loaded.assert_ready()


def test_plan_storage_outside_library_and_never_overwrites(tmp_path):
    plan = plan_fixture(tmp_path)
    with pytest.raises(PlanTransportError):
        save_plan(plan, plan.root / "plan.json")
    path = tmp_path / "state" / "plan.json"
    save_plan(plan, path)
    with pytest.raises(PlanTransportError):
        save_plan(plan, path)


@pytest.mark.skipif(os.name != "nt", reason="native Windows integration")
@pytest.mark.parametrize("entry", ["managed_worker", "namespace_lease"])
def test_execution_storage_inside_library_rejected_before_writes(tmp_path, entry):
    plan = plan_fixture(tmp_path)
    state = plan.root / "state"
    with pytest.raises(ValueError, match="outside library"):
        if entry == "managed_worker":
            ManagedWorker(plan, state, command=[sys.executable, "-c", "pass"])
        else:
            NamespaceLease(state, str(uuid4()), "Local\\invalid-storage", plan).acquire()
    assert not state.exists()


def test_e13_transcribe_only_preflight_readonly_no_sakura(tmp_path):
    plan = plan_fixture(tmp_path, mode="transcribe_only")
    assets = local_assets(tmp_path)
    before = {str(p): p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
    def request(_):
        pytest.fail("transcribe_only must not contact Sakura")
    with httpx.Client(transport=httpx.MockTransport(request)) as client:
        result = preflight(plan, assets, client, runtime_probe=lambda: True)
    assert result["available"] and not result["translation"]["required"]
    assert result["inference_proven"] is False
    assert before == {str(p): p.stat().st_mtime_ns for p in tmp_path.rglob("*")}


@pytest.mark.parametrize("backend,model", [("xxl", "large-v3-turbo"), ("kotoba", "kotoba-tech/kotoba-whisper-v2.0-faster")])
def test_e14_missing_local_assets_no_implicit_download(tmp_path, backend, model):
    plan = plan_fixture(tmp_path, mode="transcribe_only", transcribe_backend=backend, transcribe_model=model)
    result = preflight(plan, tmp_path, runtime_probe=lambda: True)
    assert not result["asr"]["available"]
    assert result["asr"]["error_code"] in {"ASR_EXECUTABLE_MISSING", "ASR_LOCAL_MODEL_REQUIRED"}
    assert list(plan.root.rglob("*.srt")) == []


def test_models_availability_is_not_inference(tmp_path):
    plan = plan_fixture(tmp_path)
    assets = local_assets(tmp_path)
    calls = []
    def request(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"data": [{"id": plan.settings["translate_model"]}]})
    with httpx.Client(transport=httpx.MockTransport(request)) as client:
        result = preflight(plan, assets, client, runtime_probe=lambda: True)
    assert result["available"] and calls == [("GET", "/v1/models")]
    assert not result["translation"]["inference_proven"]


@pytest.mark.parametrize("endpoint", ["http://user:secret@127.0.0.1/v1", "http://127.0.0.1/v1?token=secret", "file:///secret"])
def test_preflight_rejects_secret_or_invalid_endpoints(tmp_path, endpoint):
    plan = plan_fixture(tmp_path, translate_api=endpoint)
    result = preflight(plan, tmp_path)
    assert result["translation"]["error_code"] == "TRANSLATION_ENDPOINT_INVALID"
    assert "secret" not in json.dumps(result)


def start_worker(plan, tmp_path, mode="full", production=False):
    def command(worker):
        digest = save_plan(plan, worker.run_dir / "plan.json")
        prefix = [sys.executable, "-B", "-u", str(WORKER)] if production else [sys.executable, "-B", "-u", str(DRIVER), mode]
        return prefix + ["run", "--run-id", worker.run_id, "--job-name", worker.job.name,
            "--state-root", str(worker.state_root), "--project-root", str(PROJECT),
            "--plan", str(worker.run_dir / "plan.json"), "--expected-digest", digest]
    return ManagedWorker(plan, tmp_path / "state", command_factory=command).start()


@pytest.mark.skipif(os.name != "nt", reason="native Windows integration")
@pytest.mark.parametrize("mode", ["full", "transcribe_only"])
def test_isolated_worker_authority_and_private_streams(tmp_path, mode):
    plan = plan_fixture(tmp_path, mode=mode, names=("a/voice01.flac", "b/voice01 SEなし.wav", "voice02.flac"), parallel=3)
    worker = start_worker(plan, tmp_path, mode)
    try:
        cleanup = worker.wait(timeout=20)
        assert cleanup["cleanup_verified"] and cleanup["active_count"] == 0, cleanup
        assert cleanup["result"]["outcome"] == "completed", (cleanup, worker.stderr_bytes)
        assert sum(cleanup["result"]["counts"].values()) == 3
        events = JSONLDecoder(worker.run_id).feed(worker.stdout_bytes)
        assert events[-1]["event"] == "run_finished"
        assert events[-1]["payload"]["counts"]["copied"] == 1
        assert b"PRIVATE" not in worker.stdout_bytes + worker.stderr_bytes
        assert all(event["payload"].get("progress_mode") == "indeterminate" for event in events if event["event"] == "stage_changed" and event["stage"] == "translation")
        (tmp_path / "worker_evidence.json").write_text(json.dumps(cleanup), encoding="utf-8")
    finally:
        worker.close()


@pytest.mark.skipif(os.name != "nt", reason="native Windows integration")
def test_isolated_worker_failure_100percent_asr_is_not_success(tmp_path):
    plan = plan_fixture(tmp_path, names=("fail01.flac",))
    worker = start_worker(plan, tmp_path)
    try:
        result = worker.wait(timeout=20)
        assert result["cleanup_verified"]
        assert result["result"]["outcome"] == "completed_with_errors", result
        assert worker.process.returncode == 1
        assert b"PRIVATE" not in worker.stdout_bytes + worker.stderr_bytes
        assert not plan.tracks[0].task.final_srt_path.exists()
    finally:
        worker.close()


@pytest.mark.skipif(os.name != "nt", reason="native Windows integration")
def test_isolated_worker_duplicate_cancel_and_snapshot(tmp_path):
    plan = plan_fixture(tmp_path, names=("slow01.flac", "slow02.flac"), parallel=1)
    worker = start_worker(plan, tmp_path)
    try:
        deadline = time.monotonic() + 8
        while b'"event":"track_started"' not in worker.stdout_bytes:
            assert time.monotonic() < deadline, worker.stderr_bytes
            time.sleep(.01)
        snapshot = {"schema_version": "1.0", "run_id": worker.run_id, "request_id": str(uuid4()), "command": "request_snapshot", "payload": {}}
        cancel = {**snapshot, "request_id": str(uuid4()), "command": "cancel"}
        assert worker.send_control(snapshot) and worker.send_control(cancel) and worker.send_control(cancel)
        result = worker.wait(timeout=20)
        assert result["cleanup_verified"] and result["result"]["outcome"] == "cancelled", result
        events = JSONLDecoder(worker.run_id).feed(worker.stdout_bytes)
        acknowledgments = [event for event in events if event["event"] == "control_ack"]
        assert len(acknowledgments) == 2
        assert sum(result["result"]["counts"].values()) == 2
    finally:
        worker.close()


@pytest.mark.skipif(os.name != "nt", reason="native Windows integration")
def test_preflight_failure_never_starts_tracks(tmp_path):
    plan = plan_fixture(tmp_path)
    worker = start_worker(plan, tmp_path, "preflight_fail")
    try:
        result = worker.wait(timeout=20)
        assert result["cleanup_verified"], result
        assert result["result"]["outcome"] == "preflight_failed", result
        assert not result["result"]["complete"] and worker.process.returncode == 3
        assert b"track_started" not in worker.stdout_bytes
        assert not list(plan.root.rglob("*.srt"))
    finally:
        worker.close()


@pytest.mark.skipif(os.name != "nt", reason="native Windows integration")
def test_readonly_scan_and_preflight_commands(tmp_path):
    plan = plan_fixture(tmp_path, mode="transcribe_only")
    state_root = tmp_path / "state"
    before = {str(p): p.stat().st_mtime_ns for p in plan.root.rglob("*")}
    scan_id, preflight_id = str(uuid4()), str(uuid4())
    job = WindowsJob("Local\\VoiceTransl-helper-" + scan_id)
    def invoke(operation, run_id, arguments):
        command = [sys.executable, "-B", "-u", str(RUNTIME), "bootstrap", "--job-name", job.name,
            "--", sys.executable, "-B", "-u", str(WORKER), operation, "--run-id", run_id,
            "--job-name", job.name, "--state-root", str(state_root), "--project-root", str(tmp_path), *arguments]
        result = subprocess.run(command, capture_output=True, timeout=15)
        events = JSONLDecoder(run_id).feed(result.stdout)
        assert len(events) == 1 and events[0]["event"] == "worker_ready", result.stderr
        return result
    try:
        scanned = invoke("scan", scan_id, ["--root", str(plan.root), "--mode", "transcribe_only"])
        assert scanned.returncode == 0
        scan_dir = state_root / "runs" / scan_id
        document = json.loads((scan_dir / "scan.json").read_text(encoding="utf-8"))
        loaded = load_plan(scan_dir / "plan.json", document["transport_hash"])
        assert loaded == plan
        checked = invoke("preflight", preflight_id, ["--plan", str(scan_dir / "plan.json"), "--expected-digest", document["transport_hash"]])
        assert checked.returncode == 3  # local assets deliberately absent
        availability = json.loads((state_root / "runs" / preflight_id / "preflight.json").read_text(encoding="utf-8"))
        assert not availability["translation"]["required"] and not availability["asr"]["available"]
        assert before == {str(p): p.stat().st_mtime_ns for p in plan.root.rglob("*")}
        assert job.active_count() == 0
    finally:
        job.terminate()
        job.close()


@pytest.mark.skipif(os.name != "nt", reason="native Windows integration")
def test_uncontained_worker_fails_closed(tmp_path):
    plan = plan_fixture(tmp_path)
    run_id = str(uuid4())
    result = subprocess.run([sys.executable, "-B", str(WORKER), "scan", "--run-id", run_id,
        "--job-name", "Local\\VoiceTransl-absent-" + run_id, "--state-root", str(tmp_path / "state"),
        "--root", str(plan.root)], capture_output=True, timeout=10)
    assert result.returncode == 4 and result.stdout == b""
    assert result.stderr.strip() == b"ASMR_WORKER_OPERATION_FAILED"
    assert not (tmp_path / "state").exists()
    assert not list(plan.root.rglob("*.srt"))


@pytest.mark.skipif(os.name != "nt", reason="native Windows integration")
def test_invalid_control_reconciles_protocol_failure(tmp_path):
    plan = plan_fixture(tmp_path, names=("slow01.flac",))
    worker = start_worker(plan, tmp_path)
    try:
        deadline = time.monotonic() + 8
        while b'"event":"track_started"' not in worker.stdout_bytes:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert worker.send_control({"schema_version": "1.0", "run_id": str(uuid4()),
            "request_id": str(uuid4()), "command": "cancel", "payload": {}})
        result = worker.wait(timeout=20)
        assert result["cleanup_verified"] and result["result"]["outcome"] == "protocol_failed", result
        assert not result["result"]["complete"] and worker.process.returncode == 4
    finally:
        worker.close()


@pytest.mark.parametrize("native_exit", [0, 7, None])
def test_strict_native_exit_metadata_no_warning_free_failure(tmp_path, monkeypatch, native_exit):
    plan = plan_fixture(tmp_path, mode="transcribe_only")
    assets = local_assets(tmp_path)
    private = "PRIVATE_NATIVE_SUBTITLE"
    def runner(argv, cwd, callback):
        output = Path(argv[argv.index("--output_dir") + 1])
        (output / "voice01.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n" + private + "\n", encoding="utf-8")
        callback(1., 1., 1.)
        return native_exit, "PRIVATE_NATIVE_LOG"
    monkeypatch.setattr("scripts.asmr_transcribe_lib.local_faster_whisper._run_with_progress", runner)
    metadata, stream = {}, io.StringIO()
    store = RunStore(tmp_path / "run", str(uuid4()), plan, stream, asr_metadata=metadata)
    store.start()
    outcomes = run_plan(plan, project_root=assets, translator=None, opencc_converter=lambda *args: pytest.fail("not needed"),
        transcriber=strict_xxl_transcriber(plan, metadata), observer=store.observer)
    result = store.finish(outcomes)
    track = plan.tracks[0]
    receipt = read_json(store.run_dir / "receipts" / (track.track_id + ".json"))
    assert receipt["native_metadata"] == {"backend": "xxl", "native_exit_code": native_exit, "returncode_known": native_exit is not None}
    if native_exit == 0:
        assert result["outcome"] == "completed" and receipt["warnings"] == []
    else:
        assert result["outcome"] == "completed_with_errors" and result["exit_code"] == 1
        assert receipt["status"] == "failed" and receipt["warnings"]
        assert receipt["provenance"] is None
    assert private not in stream.getvalue() and "PRIVATE_NATIVE_LOG" not in json.dumps(receipt)


def test_preflight_missing_pipeline_dependency_is_structured(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path, mode="transcribe_only")
    assets = local_assets(tmp_path)
    import importlib.util
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "opencc" else original(name))
    result = preflight(plan, assets, runtime_probe=lambda: True)
    assert not result["available"] and result["runtime"]["error_code"] == "PIPELINE_DEPENDENCY_MISSING"


@pytest.mark.skipif(os.name != "nt", reason="native Windows integration")
def test_real_owned_native_exit7_with_valid_srt_is_failed(tmp_path):
    plan = plan_fixture(tmp_path, mode="transcribe_only")
    worker = start_worker(plan, tmp_path, "native_exit7")
    try:
        result = worker.wait(timeout=20)
        assert result["cleanup_verified"] and worker.process.returncode == 1, result
        assert result["result"]["outcome"] == "completed_with_errors"
        receipt = read_json(worker.run_dir / "receipts" / (plan.tracks[0].track_id + ".json"))
        assert receipt["native_metadata"]["native_exit_code"] == 7
        assert receipt["status"] == "failed" and receipt["warnings"] == ["ASR_NATIVE_NONZERO_EXIT"]
        assert plan.tracks[0].task.transcribe_srt_path.exists()  # retained, uncertified partial
        assert b"PRIVATE" not in worker.stdout_bytes + worker.stderr_bytes
        (tmp_path / "native_exit_evidence.json").write_text(json.dumps(result), encoding="utf-8")
    finally:
        worker.close()
