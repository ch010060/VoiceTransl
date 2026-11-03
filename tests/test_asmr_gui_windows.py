"""Native Windows containment evidence, using only controlled fake processes."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="requires native Windows Job Objects")
FIXTURE = Path(__file__).parent / "fixtures" / "asmr_gui_native_fixture.py"


def wait_json(path, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.02)
    raise AssertionError(f"controlled fixture did not publish {path}")


class ProcessHandles:
    """Retain native handles so PID reuse cannot make cleanup verification ambiguous."""
    def __init__(self, pids):
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        self.kernel32.OpenProcess.restype = ctypes.c_void_p
        self.kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.kernel32.WaitForSingleObject.restype = ctypes.c_ulong
        self.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.handles = {}
        for pid in pids:
            handle = self.kernel32.OpenProcess(0x100000, False, pid)
            assert handle, f"cannot retain owned PID {pid}: {ctypes.get_last_error()}"
            self.handles[pid] = handle

    def exited(self, timeout_ms=5000):
        return {pid: self.kernel32.WaitForSingleObject(handle, timeout_ms) == 0
                for pid, handle in self.handles.items()}

    def close(self):
        for handle in self.handles.values():
            self.kernel32.CloseHandle(handle)


def library_plan(tmp_path):
    from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan
    library = tmp_path / "library"
    (library / "work").mkdir(parents=True, exist_ok=True)
    (library / "work" / "fake.wav").write_bytes(b"controlled synthetic input, never decoded")
    plan = build_execution_plan(library, mode="transcribe_only")
    assert plan.executable
    return plan


def make_worker(tmp_path, mode="forced", *, plan=None):
    from scripts.asmr_gui_runtime import ManagedWorker
    directory = tmp_path / "processes"
    directory.mkdir(parents=True, exist_ok=True)
    worker = ManagedWorker(plan or library_plan(tmp_path), tmp_path / "state", [
        sys.executable, "-B", "-u", str(FIXTURE), "tree", "--directory", str(directory),
        "--job-name", "{job_name}", "--mode", mode,
    ], cooperative_timeout=0.25, cleanup_timeout=3)
    return worker, directory


def tree_pids(directory):
    wait_json(directory / "tree_ready.json")
    return {role: wait_json(directory / (role + ".json"))["pid"]
            for role in ("worker", "child", "grandchild")}


def evidence(tmp_path, scenario, *, run_id=None, pids=None, counts=None, result=None,
             elapsed=None, **extra):
    value = {"scenario": scenario, "native_windows": True,
             "run_id": run_id, "pids": pids or {}, "active_counts": counts or [],
             "budgets_seconds": {"cooperative": 0.25, "cleanup": 3},
             "measured_elapsed_seconds": elapsed,
             "cleanup_status": result, **extra}
    (tmp_path / "native_evidence.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def final_cleanup(worker):
    if worker.job is not None:
        try:
            worker.stop()
        finally:
            # Only the exact owned Job may be terminated if a test assertion failed.
            try:
                worker.job.terminate()
                worker.job.close()
            except (OSError, RuntimeError, AttributeError):
                pass


def test_w02_default_five_second_budgets_native_tree(tmp_path):
    worker, directory = make_worker(tmp_path, "forced")
    worker.cooperative_timeout = 5
    worker.cleanup_timeout = 5
    handles = None
    try:
        worker.start()
        pids = tree_pids(directory)
        handles = ProcessHandles(pids.values())
        before = worker.job.active_count()
        started = time.monotonic()
        result = worker.stop()
        elapsed = time.monotonic() - started
        assert before == 4 and result["active_count"] == 0
        assert result["cleanup_verified"] and result["forced"]
        assert 4.9 <= elapsed < 10.5, result
        exited = handles.exited()
        assert all(exited.values())
        evidence(tmp_path, "W02_default_budgets", run_id=worker.run_id, pids=pids,
                 counts=[before, 0], result=result, elapsed=elapsed,
                 budgets_seconds={"cooperative": 5, "cleanup": 5},
                 process_handles_exited=exited)
    finally:
        final_cleanup(worker)
        if handles:
            handles.close()


@pytest.mark.parametrize("mode", ["forced", "cooperative"])
def test_w01_w02_native_tree_stop(tmp_path, mode):
    worker, directory = make_worker(tmp_path, mode)
    handles = None
    try:
        worker.start()
        pids = tree_pids(directory)
        handles = ProcessHandles(pids.values())
        owned = set(worker.job.process_ids())
        before = worker.job.active_count()
        assert set(pids.values()) <= owned
        assert before >= 3
        started = time.monotonic()
        result = worker.stop()
        elapsed = time.monotonic() - started
        exited = handles.exited()
        evidence(tmp_path, "W01,W02", run_id=worker.run_id, pids=pids,
                 counts=[before, result["active_count"]], result=result, elapsed=elapsed,
                 mode=mode, owned_process_ids=sorted(owned), process_handles_exited=exited)
        assert result["cleanup_verified"] and result["active_count"] == 0
        assert result["reconciliation"]["outcome"] == "interrupted"
        assert result["reconciliation"]["complete"] is False
        assert all(exited.values())
        assert result["forced"] == (mode == "forced")
        assert elapsed < 5.5
        assert not any(path.exists() for path in worker.lease.markers)
        if mode == "cooperative":
            assert (directory / "grandchild_exited.json").exists()
    finally:
        final_cleanup(worker)
        if handles:
            handles.close()


def test_w03_worker_exits_before_descendants(tmp_path):
    worker, directory = make_worker(tmp_path, "early_exit")
    handles = None
    try:
        worker.start()
        pids = tree_pids(directory)
        handles = ProcessHandles([pids["child"], pids["grandchild"]])
        worker.process.wait(timeout=10)
        before = worker.job.active_count()
        assert before >= 2
        started = time.monotonic()
        result = worker.wait(timeout=5)
        exited = handles.exited()
        evidence(tmp_path, "W03", run_id=worker.run_id, pids=pids,
                 counts=[before, result["active_count"]], result=result,
                 elapsed=time.monotonic() - started, process_handles_exited=exited)
        assert result["cleanup_verified"] and result["forced"]
        assert all(exited.values())
        assert result["reconciliation"]["outcome"] != "completed"
    finally:
        final_cleanup(worker)
        if handles:
            handles.close()


def test_w04_native_assignment_failure_prevents_fake_work(tmp_path):
    from scripts.asmr_gui_runtime import WindowsJob
    job = WindowsJob("Local\\VoiceTransl-test-" + str(uuid.uuid4()))
    process = None
    try:
        started = time.monotonic()
        process = subprocess.Popen([sys.executable, "-B", "-u", str(FIXTURE),
            "assignment_failure", "--directory", str(tmp_path), "--job-name", job.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=10)
        count = job.active_count()
        evidence(tmp_path, "W04", run_id=str(uuid.uuid4()), pids={"bootstrap": process.pid},
                 counts=[count], result={"exit_code": process.returncode},
                 elapsed=time.monotonic() - started, native_fault="invalid_job_handle_assignment")
        assert process.returncode == 125, (stdout, stderr)
        assert count == 0
        assert not (tmp_path / "inference_started.json").exists()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        job.close()


def test_w05_repeated_stop_is_idempotent(tmp_path):
    worker, directory = make_worker(tmp_path)
    try:
        worker.start()
        pids = tree_pids(directory)
        first = worker.stop()
        started = time.monotonic()
        second = worker.stop()
        evidence(tmp_path, "W05", run_id=worker.run_id, pids=pids,
                 counts=[first["active_count"], second["active_count"]], result=second,
                 elapsed=time.monotonic() - started, first_stop=first)
        assert first["cleanup_verified"] and second["cleanup_verified"]
        assert second["state"] == first["state"]
    finally:
        final_cleanup(worker)


def test_w06_cleanup_failure_retains_namespace_and_retry(tmp_path, monkeypatch):
    from scripts.asmr_transcribe_lib.gui_ownership import NamespaceLease
    plan = library_plan(tmp_path)
    worker, directory = make_worker(tmp_path, plan=plan)
    try:
        worker.start()
        pids = tree_pids(directory)
        markers = list(worker.lease.markers)
        worker.cleanup_timeout = 0.15
        started = time.monotonic()
        with monkeypatch.context() as fault:
            fault.setattr(worker.job, "terminate", lambda *args, **kwargs: None)
            # Preserve actual native count: processes really remain alive in the owned Job.
            failed = worker.stop()
            assert failed["state"] == "cleanup_failed"
            assert not failed["cleanup_verified"]
            assert worker.job.active_count() >= 3
            assert all(path.exists() for path in markers)
            competitor = NamespaceLease(tmp_path / "state", str(uuid.uuid4()), "other-job", plan)
            with pytest.raises((RuntimeError, OSError, ValueError)):
                competitor.acquire()
        worker.cleanup_timeout = 3
        retried = worker.stop()
        evidence(tmp_path, "W06,N02,N08", run_id=worker.run_id, pids=pids,
                 counts=[failed["active_count"], retried["active_count"]], result=retried,
                 elapsed=time.monotonic() - started, injected_fault="terminate_noop",
                 initial_cleanup=failed, retained_markers=[str(p) for p in markers])
        assert retried["cleanup_verified"] and retried["active_count"] == 0
        assert not any(path.exists() for path in markers)
    finally:
        final_cleanup(worker)


@pytest.mark.parametrize("action", ["close", "crash"])
def test_w07_w10_controller_close_or_crash(tmp_path, action):
    from scripts.asmr_transcribe_lib.gui_ownership import NamespaceLease
    plan = library_plan(tmp_path)
    directory = tmp_path / "processes"
    directory.mkdir()
    run_id = str(uuid.uuid4())
    controller = subprocess.Popen([sys.executable, "-B", "-u", str(FIXTURE), "controller",
        "--directory", str(directory), "--library", str(plan.root),
        "--state-root", str(tmp_path / "state"), "--run-id", run_id],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    handles = None
    try:
        ready = wait_json(directory / "controller_ready.json")
        pids = tree_pids(directory)
        handles = ProcessHandles(pids.values())
        assert set(pids.values()) <= set(ready["owned_process_ids"])
        # The incumbent lease is held by another real controller process.
        competitor = NamespaceLease(tmp_path / "state", str(uuid.uuid4()), "competing-job", plan)
        with pytest.raises((RuntimeError, OSError, ValueError)):
            competitor.acquire()
        independent = NamespaceLease(tmp_path / "state", str(uuid.uuid4()), "independent-job",
                                     library_plan(tmp_path / "independent"))
        independent.acquire()
        independent.settle(cleanup_verified=True)  # No process was launched for this lease.
        started = time.monotonic()
        stdout, stderr = controller.communicate((action + "\n").encode(), timeout=15)
        assert controller.returncode == (0 if action == "close" else 73), (stdout, stderr)
        exited = handles.exited()
        assert all(exited.values())
        if action == "crash":
            assert all(Path(p).exists() for p in ready["markers"])
            competitor = NamespaceLease(tmp_path / "state", str(uuid.uuid4()), "restart-job", plan)
            with pytest.raises((RuntimeError, OSError, ValueError)):
                competitor.acquire()
            cleanup = {"cleanup_verified": True, "restart_blocked": True,
                       "persistent_marker_retained": True}
        else:
            cleanup = wait_json(directory / "controller_closed.json")["result"]
            assert cleanup["cleanup_verified"]
            assert not any(Path(p).exists() for p in ready["markers"])
        evidence(tmp_path, "W07,W10,N05,N06", run_id=run_id,
                 pids={"controller": controller.pid, **pids},
                 counts=[ready["active_count"], cleanup.get("active_count")],
                 result=cleanup, elapsed=time.monotonic() - started,
                 controller_action=action, process_handles_exited=exited, job_name=ready["job_name"],
                 cross_instance_conflict_rejected=True, independent_namespace_acquired=True,
                 owned_process_ids=ready["owned_process_ids"])
    finally:
        if controller.poll() is None:
            # Exact held controller process handle; Job last-handle close contains its tree.
            controller.terminate()
            controller.wait(timeout=5)
        if handles:
            handles.close()


def test_w08_w09_unrelated_dummy_and_external_http_survive(tmp_path):
    worker, directory = make_worker(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    dummy = subprocess.Popen([sys.executable, "-B", "-u", str(FIXTURE), "dummy",
                              "--directory", str(external)])
    server = subprocess.Popen([sys.executable, "-B", "-u", str(FIXTURE), "server",
                               "--directory", str(external)])
    try:
        wait_json(external / "dummy.json")
        server_info = wait_json(external / "server.json")
        url = f'http://127.0.0.1:{server_info["port"]}/models'
        with urllib.request.urlopen(url, timeout=3) as response:
            assert json.load(response)["data"][0]["id"] == "synthetic-sakura"
        worker.start()
        pids = tree_pids(directory)
        owned = set(worker.job.process_ids())
        assert dummy.pid not in owned and server.pid not in owned
        started = time.monotonic()
        result = worker.stop()
        assert dummy.poll() is None and server.poll() is None
        with urllib.request.urlopen(url, timeout=3) as response:
            reachable = response.status == 200
        evidence(tmp_path, "W08,W09", run_id=worker.run_id, pids=pids,
                 counts=[len(owned), result["active_count"]], result=result,
                 elapsed=time.monotonic() - started,
                 unrelated_pids={"dummy": dummy.pid, "external_sakura_like_http": server.pid},
                 unrelated_survived=True, external_models_reachable=reachable)
        assert result["cleanup_verified"] and reachable
    finally:
        final_cleanup(worker)
        # These exact Popen handles were created here and never belonged to the owned Job.
        for process in (dummy, server):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
