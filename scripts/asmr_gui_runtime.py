"""Controller-owned Windows containment for the isolated ASMR worker.

Only the controller retains the kill-on-close Job handle. The bootstrap joins
that Job before creating the worker; no executable supplied by the caller runs
before assignment succeeds. External model servers are never added to the Job.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import importlib
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid


class ContainmentError(RuntimeError):
    pass


def _windows():
    if os.name != "nt":
        raise ContainmentError("Windows Job Objects require Windows")
    return _KERNEL


class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                 "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IOCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in
                ("TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")] + [
                    (name, wintypes.DWORD) for name in
                    ("TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]


if os.name == "nt":
    _KERNEL = ctypes.WinDLL("kernel32", use_last_error=True)
    for _name, _args, _result in [
        ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        ("OpenJobObjectW", [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR], wintypes.HANDLE),
        ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        ("QueryInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
        ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        ("IsProcessInJob", [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)], wintypes.BOOL),
        ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ("SetHandleInformation", [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD], wintypes.BOOL),
        ("GetCurrentProcess", [], wintypes.HANDLE),
    ]:
        _fn = getattr(_KERNEL, _name)
        _fn.argtypes, _fn.restype = _args, _result


def _check(ok):
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return ok


def _library(name):
    prefix = "scripts." if __package__ == "scripts" else ""
    return importlib.import_module(prefix + "asmr_transcribe_lib." + name)


class WindowsJob:
    def __init__(self, name: str, create: bool = True):
        kernel = _windows()
        self.name, self.handle = name, None
        ctypes.set_last_error(0)
        handle = (kernel.CreateJobObjectW(None, name) if create else
                  kernel.OpenJobObjectW(0x0001 | 0x0004 | 0x0008, False, name))
        error = ctypes.get_last_error()
        if not handle:
            raise ctypes.WinError(error)
        self.handle = handle
        try:
            if create and error == 183:
                raise ContainmentError("owned Job name already exists")
            _check(kernel.SetHandleInformation(handle, 1, 0))
            if create:
                limits = _ExtendedLimits()
                # Deliberately omit BREAKAWAY_OK and SILENT_BREAKAWAY_OK.
                limits.BasicLimitInformation.LimitFlags = 0x00002000
                _check(kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    def assign_current(self):
        kernel = _windows()
        _check(kernel.AssignProcessToJobObject(self.handle, kernel.GetCurrentProcess()))
        self.assert_current()

    def assert_current(self):
        kernel, member = _windows(), wintypes.BOOL()
        _check(kernel.IsProcessInJob(kernel.GetCurrentProcess(), self.handle, ctypes.byref(member)))
        if not member.value:
            raise ContainmentError("worker is not contained in the expected Job")

    def active_count(self) -> int:
        info = _Accounting()
        _check(_windows().QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None))
        return int(info.ActiveProcesses)

    def process_ids(self) -> list[int]:
        capacity = 32
        while capacity <= 1048576:
            class ProcessList(ctypes.Structure):
                _fields_ = [("assigned", wintypes.DWORD), ("count", wintypes.DWORD),
                            ("ids", ctypes.c_size_t * capacity)]
            info = ProcessList()
            if _windows().QueryInformationJobObject(self.handle, 3, ctypes.byref(info), ctypes.sizeof(info), None):
                return list(info.ids[:info.count])
            if ctypes.get_last_error() != 234:
                raise ctypes.WinError(ctypes.get_last_error())
            capacity = max(capacity * 2, info.assigned + 16)
        raise ContainmentError("Job process list exceeded supported bound")

    def terminate(self, exit_code: int = 1):
        _check(_windows().TerminateJobObject(self.handle, exit_code))

    def close(self):
        if self.handle is not None:
            _check(_windows().CloseHandle(self.handle))
            self.handle = None


def assert_current_in_job(name: str):
    job = WindowsJob(name, create=False)
    try:
        job.assert_current()
    finally:
        job.close()


def bootstrap(job_name: str, command: list[str]) -> int:
    try:
        job = WindowsJob(job_name, create=False)
        try:
            job.assign_current()
        finally:
            # Do not keep the controller's Job alive after controller crash.
            job.close()
    except (OSError, ContainmentError):
        print("ASMR_CONTAINMENT_FAILED", file=sys.stderr, flush=True)
        return 125
    try:
        child = subprocess.Popen(command, stdin=sys.stdin.buffer, stdout=sys.stdout.buffer,
                                 stderr=sys.stderr.buffer, close_fds=True)
        return child.wait()
    except OSError:
        print("ASMR_WORKER_START_FAILED", file=sys.stderr, flush=True)
        return 125


class ManagedWorker:
    """Synchronous lifecycle facade; callers may invoke wait/stop off the UI thread.

    Readers and control writes run in daemon threads, so inherited stdio or a
    worker that ignores stdin cannot turn a bounded Stop into an unbounded wait.
    Captured output is capped at 1 MiB per stream; on_stdout receives all bytes.
    """
    def __init__(self, plan, state_root, command=None, run_id=None,
                 cooperative_timeout=5, cleanup_timeout=5, *, command_factory=None,
                 on_stdout=None):
        _windows()
        self.plan, self.state_root = plan, Path(state_root).resolve()
        if plan is not None and self.state_root.is_relative_to(Path(plan.root).resolve()):
            raise ValueError("state root must be outside library")
        self.run_id = str(uuid.UUID(run_id)) if run_id else str(uuid.uuid4())
        self.run_dir = self.state_root / "runs" / self.run_id
        self.command, self.command_factory, self.on_stdout = command, command_factory, on_stdout
        self.cooperative_timeout, self.cleanup_timeout = float(cooperative_timeout), float(cleanup_timeout)
        if self.cooperative_timeout < 0 or self.cleanup_timeout < 0:
            raise ValueError("cleanup budgets must be nonnegative")
        self.job = self.lease = self.process = None
        self.state = "created"
        self.stdout_chunks, self.stderr_chunks = [], []
        self._stream_lock, self._lifecycle_lock = threading.Lock(), threading.RLock()
        self._controls = queue.Queue(maxsize=16)
        self._threads = []
        self._result = None
        self._known_pids = set()

    @property
    def stdout_bytes(self):
        with self._stream_lock:
            return b"".join(self.stdout_chunks)

    @property
    def stderr_bytes(self):
        with self._stream_lock:
            return b"".join(self.stderr_chunks)

    def _drain(self, stream, chunks, callback=None):
        retained = 0
        try:
            while True:
                data = stream.read1(65536)
                if not data:
                    break
                with self._stream_lock:
                    if retained < 1048576:
                        piece = data[:1048576 - retained]
                        chunks.append(piece)
                        retained += len(piece)
                if callback:
                    try:
                        callback(data)
                    except Exception:
                        # Consumer failure cannot stop pipe draining/cleanup.
                        pass
        except (OSError, ValueError):
            pass
        finally:
            stream.close()

    def _write_controls(self):
        try:
            while True:
                value = self._controls.get()
                if value is None:
                    return
                self.process.stdin.write(value)
                self.process.stdin.flush()
        except (OSError, ValueError):
            pass
        finally:
            self.process.stdin.close()

    def _prepare_run(self):
        NamespaceLease = _library("gui_ownership").NamespaceLease
        self.lease = NamespaceLease(self.state_root, self.run_id, self.job.name, self.plan)
        self.lease.acquire()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _reconcile_cleanup(self, *, cancel, forced, root_exited_on_entry):
        reconciliation = _library("gui_ownership").reconcile_after_cleanup(
            self.run_dir, self.run_id, self.plan.plan_hash, self.process.returncode,
            [track.track_id for track in getattr(self.plan, "planned_tracks", ())])
        if forced and (not cancel or root_exited_on_entry):
            reconciliation = dict(reconciliation, outcome="interrupted", complete=False,
                                  error="OWNED_DESCENDANTS_OUTLIVED_WORKER")
        return reconciliation

    def start(self):
        with self._lifecycle_lock:
            if self.state != "created":
                raise ContainmentError("worker has already been started")
            self.job = WindowsJob("Local\\VoiceTransl-ASMR-" + self.run_id)
            try:
                self._prepare_run()
                argv = self.command_factory(self) if self.command_factory else self.command
                if not argv or isinstance(argv, (str, bytes)):
                    raise ValueError("worker command must be a nonempty argv sequence")
                argv = [str(item).replace("{job_name}", self.job.name) for item in argv]
                self.process = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "bootstrap",
                     "--job-name", self.job.name, "--", *argv],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    close_fds=True, cwd=str(Path(__file__).resolve().parent.parent))
                self.state = "running"
                for target, args in [(self._drain, (self.process.stdout, self.stdout_chunks, self.on_stdout)),
                                     (self._drain, (self.process.stderr, self.stderr_chunks)),
                                     (self._write_controls, ())]:
                    thread = threading.Thread(target=target, args=args, daemon=True)
                    thread.start()
                    self._threads.append(thread)
                return self
            except BaseException:
                if self.process is None:
                    # No bootstrap or worker has been created; there can be no writer.
                    if self.lease is not None and self.lease.acquired:
                        self.lease.settle(cleanup_verified=True)
                    self.job.close()
                    self.state = "start_failed"
                raise

    def poll(self):
        return self.process.poll() if self.process else None

    def send_control(self, message: dict):
        if not self.process or self.process.poll() is not None:
            return False
        data = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(data) > 65536:
            raise ValueError("control message exceeds maximum size")
        try:
            self._controls.put_nowait(data)
            return True
        except queue.Full:
            return False

    def _sample(self):
        self._known_pids.update(self.job.process_ids())
        return self.job.active_count()

    def _finish_cleanup(self, *, cancel: bool):
        atomic_record = _library("gui_ownership").atomic_record
        started, forced = time.monotonic(), False
        root_exited_on_entry = self.process.poll() is not None
        try:
            active = self._sample()
            if cancel and self.process.poll() is None:
                self.state = "cancellation_requested"
                self.send_control({"schema_version": "1.0", "run_id": self.run_id,
                                   "request_id": str(uuid.uuid4()), "command": "cancel", "payload": {"reason": "user_stop"}})
                deadline = time.monotonic() + self.cooperative_timeout
                while time.monotonic() < deadline:
                    active = self._sample()
                    if active == 0 and self.process.poll() is not None:
                        break
                    time.sleep(min(.02, max(0, deadline - time.monotonic())))
            active = self._sample()
            if active:
                self.state, forced = "stopping", True
                self.job.terminate()
            deadline = time.monotonic() + self.cleanup_timeout
            while True:
                active = self._sample()
                if active == 0 and self.process.poll() is not None:
                    break
                if time.monotonic() >= deadline:
                    raise ContainmentError("owned Job cleanup could not be verified within budget")
                time.sleep(min(.02, max(0, deadline - time.monotonic())))
            self.state = "reconciling"
            # Reconciliation is deliberately after native emptiness verification.
            reconciliation = self._reconcile_cleanup(
                cancel=cancel, forced=forced, root_exited_on_entry=root_exited_on_entry)
            result = {"schema_version": "1.0", "run_id": self.run_id, "job_name": self.job.name,
                      "plan_hash": getattr(self.plan, "plan_hash", None),
                      "state": "stopped", "cleanup_verified": True, "forced": forced,
                      "root_pid": self.process.pid, "pids": sorted(self._known_pids),
                      "active_count": 0, "exit_code": self.process.returncode,
                      "elapsed_seconds": time.monotonic() - started, "reconciliation": reconciliation,
                      "result": reconciliation}
            atomic_record(self.run_dir / "recovery.json", reconciliation)
            atomic_record(self.run_dir / "cleanup.json", result)
            if self.lease is not None:
                self.lease.settle(cleanup_verified=True)
            self.job.close()
            self.state = "stopped"
            self._result = result
            try:
                self._controls.put_nowait(None)
            except queue.Full:
                pass
            for thread in self._threads:
                thread.join(timeout=.1)
            return result
        except Exception as exc:
            self.state = "cleanup_failed"
            try:
                active = self.job.active_count()
            except Exception:
                active = None
            # Keep every ownership handle and marker so a retry can prove cleanup.
            self._result = {"state": self.state, "cleanup_verified": False, "forced": forced,
                            "run_id": self.run_id, "root_pid": self.process.pid,
                            "active_count": active,
                            "pids": sorted(self._known_pids), "error": type(exc).__name__,
                            "elapsed_seconds": time.monotonic() - started}
            return self._result

    def wait(self, timeout=None):
        if not self.process:
            raise ContainmentError("worker has not started")
        self.process.wait(timeout=timeout)
        with self._lifecycle_lock:
            if self.state == "stopped":
                return self._result
            return self._finish_cleanup(cancel=False)

    def stop(self):
        with self._lifecycle_lock:
            if self.state == "stopped":
                return self._result
            if not self.process:
                return {"state": self.state, "cleanup_verified": self.state in {"created", "start_failed"}}
            return self._finish_cleanup(cancel=True)

    def close(self):
        return self.stop()


class ManagedOperation(ManagedWorker):
    """Read-only scan/preflight helper using the same owned process lifecycle.

    Helpers reserve an operation directory and have no output NamespaceLease.
    Their artifacts describe discovery/availability, never BatchRun completion.
    library_root is required so directory reservation itself cannot write into
    the audio library before the worker has a chance to validate its arguments.
    """
    def __init__(self, operation, state_root, command_factory, *, library_root,
                 run_id=None, cooperative_timeout=5, cleanup_timeout=5,
                 on_stdout=None, expected_plan_hash=None):
        if operation not in {"scan", "preflight"}:
            raise ValueError("managed operation must be scan or preflight")
        if not callable(command_factory):
            raise ValueError("managed operation requires a command factory")
        self.operation = operation
        self.library_root = Path(library_root).resolve()
        self.expected_plan_hash = expected_plan_hash
        if Path(state_root).resolve().is_relative_to(self.library_root):
            raise ValueError("state root must be outside library")
        super().__init__(None, state_root, run_id=run_id,
                         cooperative_timeout=cooperative_timeout, cleanup_timeout=cleanup_timeout,
                         command_factory=command_factory, on_stdout=on_stdout)

    def _prepare_run(self):
        if self.state_root.resolve().is_relative_to(self.library_root.resolve()):
            raise ValueError("state root must be outside library")
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def _reconcile_cleanup(self, *, cancel, forced, root_exited_on_entry):
        result = {"schema_version": "1.0", "run_id": self.run_id, "operation": self.operation,
                  "outcome": "operation_failed", "complete": False,
                  "exit_code": self.process.returncode}
        if cancel and not root_exited_on_entry:
            return dict(result, outcome="operation_cancelled")
        if forced:
            return dict(result, error="OWNED_DESCENDANTS_OUTLIVED_HELPER")
        try:
            storage = _library("gui_results")
            artifact = storage.read_json(storage.safe_ref(self.run_dir, self.operation + ".json"))
            if (artifact.get("schema_version") != "1.0" or artifact.get("run_id") != self.run_id
                    or artifact.get("operation") != self.operation):
                raise ValueError("helper artifact identity mismatch")
            plan_hash = artifact.get("plan_hash")
            if (not isinstance(plan_hash, str) or len(plan_hash) != 64
                    or any(character not in "0123456789abcdef" for character in plan_hash)
                    or self.expected_plan_hash is not None and plan_hash != self.expected_plan_hash):
                raise ValueError("helper plan identity mismatch")
            if self.operation == "scan":
                transport_hash = artifact.get("transport_hash")
                if (not isinstance(transport_hash, str) or len(transport_hash) != 64
                        or any(character not in "0123456789abcdef" for character in transport_hash)):
                    raise ValueError("scan transport digest missing or invalid")
                plan = _library("gui_plan_io").load_plan(
                    storage.safe_ref(self.run_dir, "plan.json"), transport_hash)
                if (plan.plan_hash != plan_hash or plan.root.resolve() != self.library_root
                        or artifact.get("work_ids") != [work.work_id for work in plan.works]
                        or artifact.get("track_ids") != [track.track_id for track in plan.planned_tracks]
                        or artifact.get("issue_codes") != [issue.code for issue in plan.issues]
                        or type(artifact.get("executable")) is not bool
                        or artifact["executable"] != plan.executable):
                    raise ValueError("scan artifact differs from frozen plan")
                available = artifact["executable"]
            else:
                stages = [artifact.get(name) for name in ("asr", "translation", "conversion", "runtime")]
                if (type(artifact.get("available")) is not bool or artifact.get("inference_proven") is not False
                        or any(not isinstance(stage, dict) or type(stage.get("available")) is not bool
                               or type(stage.get("required")) is not bool for stage in stages)):
                    raise ValueError("invalid preflight availability artifact")
                if any(artifact[name].get("inference_proven") is not False for name in ("asr", "translation")):
                    raise ValueError("preflight must not certify inference")
                available = artifact["available"]
                if available != all(stage["available"] for stage in stages):
                    raise ValueError("inconsistent preflight availability")
            if self.process.returncode != (0 if available else 3):
                raise ValueError("helper exit contradicts availability")
            return dict(result, outcome="operation_completed" if available else "operation_failed",
                        complete=True, available=available, plan_hash=plan_hash,
                        artifact_ref=self.operation + ".json")
        except Exception as exc:
            return dict(result, error="HELPER_ARTIFACT_INVALID", error_type=type(exc).__name__)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    separator = argv.index("--") if "--" in argv else len(argv)
    command = argv[separator + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["bootstrap"])
    parser.add_argument("--job-name", required=True)
    args = parser.parse_args(argv[:separator])
    if not command:
        parser.error("bootstrap requires a worker command")
    return bootstrap(args.job_name, command)


if __name__ == "__main__":
    raise SystemExit(main())
