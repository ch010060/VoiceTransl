"""Native Windows LockFileEx tests; no model/backend execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from asmr_gui_runtime import ManagedOperation, ManagedWorker, WindowsJob
from asmr_transcribe_lib.gui_ownership import NamespaceLease, OwnershipConflict, verify_worker_ownership


pytestmark = pytest.mark.skipif(os.name != "nt", reason="requires native Windows LockFileEx")


def plan(*uses):
    return SimpleNamespace(plan_hash="a" * 64, output_ownership=tuple(
        SimpleNamespace(path=path, access=access, role="final") for path, access in uses))


def lease(root, candidate, job="Local\\ownership-test"):
    return NamespaceLease(root, str(uuid.uuid4()), job, candidate)


def test_n01_n02_n08_exclusive_owner_and_verified_settlement(tmp_path):
    candidate = plan((tmp_path / "out.srt", "write"))
    first, second = lease(tmp_path, candidate), lease(tmp_path, candidate)
    first.acquire()
    try:
        assert all(path.exists() for path in first.markers)
        with pytest.raises(OwnershipConflict) as error:
            second.acquire()
        assert "out.srt" not in str(error.value)
        with pytest.raises(RuntimeError, match="verified"):
            first.settle(False)
        assert first.acquired
    finally:
        first.settle(True)
    assert not any(path.exists() for path in first.markers)
    second.acquire().settle(True)


def test_n03_independent_outputs(tmp_path):
    first = lease(tmp_path, plan((tmp_path / "a.srt", "write"))).acquire()
    second = lease(tmp_path, plan((tmp_path / "b.srt", "write"))).acquire()
    first.settle(True)
    second.settle(True)


def test_n04_shared_reads_and_internal_representative_dependency(tmp_path):
    output = tmp_path / "representative.srt"
    first = lease(tmp_path, plan((output, "read"))).acquire()
    second = lease(tmp_path, plan((output, "read"))).acquire()
    try:
        with pytest.raises(OwnershipConflict):
            lease(tmp_path, plan((output, "write"))).acquire()
    finally:
        first.settle(True)
        second.settle(True)
    combined = lease(tmp_path, plan((output, "write"), (output, "read"))).acquire()
    assert len(combined.resources) == 1
    combined.settle(True)


def test_n05_crashed_process_releases_kernel_lock_but_marker_blocks(tmp_path):
    run_id = str(uuid.uuid4())
    code = """
import os,sys
from types import SimpleNamespace as S
from asmr_transcribe_lib.gui_ownership import NamespaceLease
p=S(plan_hash='a'*64,output_ownership=(S(path=sys.argv[2],role='final',access='write'),))
l=NamespaceLease(sys.argv[1],sys.argv[3],'Local\\\\crashed-ownership',p).acquire()
print('ready',flush=True)
sys.stdin.readline()
os._exit(73)
"""
    process = subprocess.Popen([sys.executable, "-u", "-c", code, str(tmp_path),
                                str(tmp_path / "out.srt"), run_id],
                               cwd=str(Path(__file__).resolve().parents[1] / "scripts"),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdout.readline() == b"ready\r\n"
        with pytest.raises(OwnershipConflict):
            lease(tmp_path, plan((tmp_path / "out.srt", "write"))).acquire()
        process.stdin.write(b"crash\n")
        process.stdin.flush()
        assert process.wait(timeout=5) == 73
        markers = list((tmp_path / "ownership").rglob("*.json"))
        assert markers
        with pytest.raises(OwnershipConflict):
            lease(tmp_path, plan((tmp_path / "out.srt", "write"))).acquire()
        with pytest.raises(OwnershipConflict):
            NamespaceLease.recover_marker(markers[0])
        (tmp_path / "native_ownership_evidence.json").write_text(json.dumps({
            "pid": process.pid, "exit_code": process.returncode,
            "unsettled_marker_count": len(markers), "unsafe_takeover_blocked": True}), encoding="utf-8")
    finally:
        if process.poll() is None:
            process.stdin.write(b"crash\n")
            process.stdin.flush()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def test_n06_unknown_job_marker_never_automatically_stolen(tmp_path):
    first = lease(tmp_path, plan((tmp_path / "out.srt", "write"))).acquire()
    first._release()  # Equivalent to OS lock release on controller crash.
    with pytest.raises(OwnershipConflict):
        lease(tmp_path, plan((tmp_path / "out.srt", "write"))).acquire()
    with pytest.raises(OwnershipConflict):
        NamespaceLease.recover_marker(first.markers[0])
    assert first.markers[0].exists()
    first.settle(True)


def test_n06_live_descendants_block_unsettled_recovery_after_lock_loss(tmp_path):
    from asmr_transcribe_lib.gui_plan import build_execution_plan
    library = tmp_path / "library"
    (library / "work").mkdir(parents=True)
    (library / "work" / "voice01.wav").write_bytes(b"controlled synthetic input, never decoded")
    candidate = build_execution_plan(library, mode="transcribe_only")
    assert candidate.executable
    directory = tmp_path / "processes"
    fixture = Path(__file__).parent / "fixtures" / "asmr_gui_native_fixture.py"
    worker = ManagedWorker(candidate, tmp_path / "state", [
        sys.executable, "-B", "-u", str(fixture), "tree", "--directory", str(directory),
        "--job-name", "{job_name}", "--mode", "forced"],
        cooperative_timeout=.1, cleanup_timeout=3).start()
    try:
        deadline = time.monotonic() + 10
        while not (directory / "tree_ready.json").exists():
            assert time.monotonic() < deadline, worker.stderr_bytes
            time.sleep(.02)
        pids = {role: json.loads((directory / (role + ".json")).read_text(encoding="utf-8"))["pid"]
                for role in ("worker", "child", "grandchild")}
        owned_before = worker.job.process_ids()
        active_before = worker.job.active_count()
        assert set(pids.values()) <= set(owned_before) and active_before >= 4
        markers = list(worker.lease.markers)
        # Simulate loss of controller kernel locks while the original native Job
        # and its worker/child/grandchild are still demonstrably alive.
        worker.lease._release()
        competitor = lease(tmp_path / "state", candidate, "Local\\competing-live-recovery")
        with pytest.raises(OwnershipConflict):
            competitor.acquire()
        with pytest.raises(OwnershipConflict):
            NamespaceLease.recover_marker(markers[0])
        active_after_recovery = worker.job.active_count()
        assert active_after_recovery >= 4
        assert set(pids.values()) <= set(worker.job.process_ids())
        assert all(marker.exists() for marker in markers)
        started = time.monotonic()
        stopped = worker.stop()
        elapsed = time.monotonic() - started
        evidence = {"scenario": "N06_live_descendants_after_kernel_lock_loss", "native_windows": True,
                    "run_id": worker.run_id, "job_name": worker.job.name,
                    "root_pid": worker.process.pid, "pids": pids, "owned_process_ids": owned_before,
                    "active_counts": [active_before, active_after_recovery, stopped.get("active_count")],
                    "competing_acquire_blocked": True, "exact_job_recovery_blocked": True,
                    "persistent_markers_retained_while_alive": True,
                    "measured_stop_seconds": elapsed, "cleanup": stopped}
        (tmp_path / "native_ownership_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        assert stopped["cleanup_verified"] is True and stopped["active_count"] == 0
        assert worker.process.poll() is not None
        assert not any(marker.exists() for marker in markers)
        competitor.acquire().settle(True)  # Prior owned writers are now verified gone.
    finally:
        worker.close()


def test_n07_partial_acquisition_rolls_back_prior_lock(tmp_path):
    outputs = [tmp_path / "a.srt", tmp_path / "b.srt"]
    probe = lease(tmp_path, plan(*[(p, "write") for p in outputs]))
    # Occupy the resource acquired second, guaranteeing a partial acquisition.
    by_resource = {lease(tmp_path, plan((p, "write"))).resources[0][0]: p for p in outputs}
    last = by_resource[probe.resources[-1][0]]
    first = by_resource[probe.resources[0][0]]
    blocker = lease(tmp_path, plan((last, "write"))).acquire()
    try:
        with pytest.raises(OwnershipConflict):
            probe.acquire()
        assert not probe._locks and not probe.markers
        unrelated = lease(tmp_path, plan((first, "write"))).acquire()
        unrelated.settle(True)
    finally:
        blocker.settle(True)


def test_explicit_recovery_requires_original_empty_job(tmp_path):
    job = WindowsJob("Local\\VoiceTransl-recover-" + str(uuid.uuid4()))
    first = lease(tmp_path, plan((tmp_path / "out.srt", "write")), job.name).acquire()
    try:
        first._release()
        NamespaceLease.recover_marker(first.markers[0])
        assert not first.markers[0].exists()
        run_dir = tmp_path / "runs" / first.run_id
        assert json.loads((run_dir / "recovery.json").read_text(encoding="utf-8"))["outcome"] == "interrupted"
        cleanup = json.loads((run_dir / "cleanup.json").read_text(encoding="utf-8"))
        assert cleanup["cleanup_verified"] is True and cleanup["active_count"] == 0
        lease(tmp_path, plan((tmp_path / "out.srt", "write"))).acquire().settle(True)
    finally:
        first.settle(True)
        job.close()


def test_worker_requires_every_exact_controller_marker(tmp_path):
    candidate = plan((tmp_path / "out.srt", "write"), (tmp_path / "source.srt", "read"))
    owner = lease(tmp_path, candidate).acquire()
    try:
        verify_worker_ownership(tmp_path, owner.run_id, owner.job_name, candidate)
        with pytest.raises(OwnershipConflict):
            verify_worker_ownership(tmp_path, str(uuid.uuid4()), owner.job_name, candidate)
        with pytest.raises(OwnershipConflict):
            verify_worker_ownership(tmp_path, owner.run_id, "Local\\foreign-job", candidate)
        original = owner.markers[0].read_text(encoding="utf-8")
        changed = json.loads(original)
        changed["plan_hash"] = "b" * 64
        owner.markers[0].write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(OwnershipConflict):
            verify_worker_ownership(tmp_path, owner.run_id, owner.job_name, candidate)
        owner.markers[0].write_text(original, encoding="utf-8")
        owner.markers[-1].unlink()
        with pytest.raises(OwnershipConflict):
            verify_worker_ownership(tmp_path, owner.run_id, owner.job_name, candidate)
    finally:
        owner.settle(True)


def test_managed_operation_rejects_state_inside_library_before_reservation(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    with pytest.raises(ValueError, match="outside library"):
        ManagedOperation("scan", library / "state", lambda worker: [], library_root=library)
    assert list(library.iterdir()) == []


def test_managed_preflight_accepts_structured_runtime_dependency_failure(tmp_path):
    helper = ManagedOperation("preflight", tmp_path / "state", lambda worker: [],
                              library_root=tmp_path / "library", expected_plan_hash="a" * 64)
    helper.run_dir.mkdir(parents=True)
    helper.process = SimpleNamespace(returncode=3)
    artifact = {"schema_version": "1.0", "run_id": helper.run_id, "operation": "preflight",
                "plan_hash": "a" * 64, "available": False, "inference_proven": False,
                "asr": {"required": True, "available": True, "inference_proven": False},
                "translation": {"required": False, "available": True, "inference_proven": False},
                "conversion": {"required": False, "available": True},
                "runtime": {"required": True, "available": False, "error_code": "PIPELINE_DEPENDENCY_MISSING"}}
    (helper.run_dir / "preflight.json").write_text(json.dumps(artifact), encoding="utf-8")
    result = helper._reconcile_cleanup(cancel=False, forced=False, root_exited_on_entry=True)
    assert result["complete"] is True and result["outcome"] == "operation_failed"
    assert "error" not in result


def test_managed_scan_and_preflight_have_owned_lifecycle_without_write_lease(tmp_path):
    library = tmp_path / "library"
    work = library / "work"
    work.mkdir(parents=True)
    (work / "voice01.flac").write_bytes(b"synthetic-not-inference")
    before = {str(path): path.stat().st_mtime_ns for path in library.rglob("*")}
    worker_path = Path(__file__).resolve().parents[1] / "scripts" / "asmr_gui_worker.py"
    state_root = tmp_path / "state"
    def command(helper, arguments):
        return [sys.executable, "-B", "-u", str(worker_path), helper.operation,
                "--run-id", helper.run_id, "--job-name", helper.job.name,
                "--state-root", str(helper.state_root), "--project-root", str(tmp_path), *arguments]
    scan = ManagedOperation("scan", state_root,
                            lambda helper: command(helper, ["--root", str(library), "--mode", "transcribe_only"]),
                            library_root=library).start()
    try:
        scanned = scan.wait(timeout=10)
        assert scanned["cleanup_verified"] is True, (scanned, scan.stderr_bytes)
        assert scanned["result"]["outcome"] == "operation_completed", (scanned, scan.stderr_bytes)
        assert scan.lease is None and not (state_root / "ownership").exists()
        artifact = json.loads((scan.run_dir / "scan.json").read_text(encoding="utf-8"))
        checked = ManagedOperation("preflight", state_root,
            lambda helper: command(helper, ["--plan", str(scan.run_dir / "plan.json"),
                                            "--expected-digest", artifact["transport_hash"]]),
            library_root=library, expected_plan_hash=artifact["plan_hash"]).start()
        try:
            available = checked.wait(timeout=10)
            assert available["cleanup_verified"] is True and available["active_count"] == 0
            assert available["result"]["outcome"] == "operation_failed"
            assert available["result"]["complete"] is True and available["result"]["available"] is False
            assert checked.lease is None
            assert "tracks" not in available["result"] and "counts" not in available["result"]
            assert before == {str(path): path.stat().st_mtime_ns for path in library.rglob("*")}
            (tmp_path / "native_helper_evidence.json").write_text(
                json.dumps({"scan": scanned, "preflight": available}), encoding="utf-8")
        finally:
            checked.close()
    finally:
        scan.close()


@pytest.mark.parametrize("corruption", ["stale_run", "schema", "exit", "inference"])
def test_managed_operation_rejects_invalid_or_contradictory_artifacts(tmp_path, corruption):
    code = """
import json,sys
from pathlib import Path
r={"schema_version":"1.0","run_id":sys.argv[2],"operation":"preflight","plan_hash":"a"*64,
   "available":True,"inference_proven":False,
   "asr":{"required":False,"available":True,"inference_proven":False},
   "translation":{"required":False,"available":True,"inference_proven":False},
   "conversion":{"required":False,"available":True}}
if sys.argv[3]=='stale_run': r['run_id']='00000000-0000-0000-0000-000000000000'
if sys.argv[3]=='schema': r['schema_version']='2.0'
if sys.argv[3]=='inference': r['asr']['inference_proven']=True
Path(sys.argv[1],'preflight.json').write_text(json.dumps(r),encoding='utf-8')
sys.exit(3 if sys.argv[3]=='exit' else 0)
"""
    helper = ManagedOperation("preflight", tmp_path / "state",
        lambda worker: [sys.executable, "-B", "-c", code, str(worker.run_dir), worker.run_id, corruption],
        library_root=tmp_path / "library", expected_plan_hash="a" * 64).start()
    try:
        result = helper.wait(timeout=10)
        assert result["cleanup_verified"] is True
        assert result["result"]["outcome"] == "operation_failed"
        assert result["result"]["error"] == "HELPER_ARTIFACT_INVALID"
        assert result["result"]["complete"] is False
    finally:
        helper.close()


def test_managed_operation_stop_retains_job_on_failure_and_retries(tmp_path):
    ready = tmp_path / "ready.json"
    code = "import json,os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid()})); time.sleep(60)"
    helper = ManagedOperation("scan", tmp_path / "state",
        lambda worker: [sys.executable, "-B", "-c", code, str(ready)],
        library_root=tmp_path / "library", cooperative_timeout=.03, cleanup_timeout=.03).start()
    native_count, native_terminate = helper.job.active_count, helper.job.terminate
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        helper.job.terminate = lambda exit_code=1: None
        failed = helper.stop()
        assert failed["cleanup_verified"] is False and failed["state"] == "cleanup_failed"
        assert helper.job.handle is not None and native_count() >= 2
        helper.job.terminate = native_terminate
        helper.cleanup_timeout = 2
        stopped = helper.stop()
        assert stopped["cleanup_verified"] is True and stopped["active_count"] == 0
        assert stopped["result"]["outcome"] == "operation_cancelled"
        assert helper.close() == stopped
        (tmp_path / "native_helper_evidence.json").write_text(json.dumps(stopped), encoding="utf-8")
    finally:
        helper.job.terminate = native_terminate
        helper.cleanup_timeout = 2
        helper.close()
