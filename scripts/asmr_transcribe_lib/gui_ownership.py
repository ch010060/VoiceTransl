"""Fail-closed Windows byte-range leases and durable unsettled ownership.

Lock availability alone never clears a marker. Explicit recovery needs the
original named Job empty, or the controller's matching verified cleanup receipt.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import uuid


class OwnershipConflict(RuntimeError):
    def __init__(self, resource_id: str):
        self.resource_id = resource_id
        super().__init__("output namespace is owned or unsettled: " + resource_id)


def atomic_record(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reconcile_after_cleanup(run_dir, run_id, plan_hash, exit_code=None, missing_track_ids=()):
    """Publishable recovery view, for callers that already proved owned writes ended."""
    prefix = "scripts." if __package__.startswith("scripts.") else ""
    reconcile = importlib.import_module(prefix + "asmr_transcribe_lib.gui_results").reconcile_run
    try:
        return reconcile(run_dir, run_id, plan_hash, exit_code=exit_code)
    except Exception as exc:
        # Keep the worker's original ledger/receipts untouched. A missing first
        # snapshot and corrupt authority both prohibit a successful result.
        return {"schema_version": "1.0", "run_id": run_id, "plan_hash": plan_hash,
                "outcome": "interrupted", "complete": False, "tracks": {}, "counts": {},
                "missing_track_ids": list(missing_track_ids), "exit_code": exit_code,
                "error": "MISSING_RUN_AUTHORITY" if isinstance(exc, FileNotFoundError)
                         else "RECONCILIATION_FAILED", "error_type": type(exc).__name__}


class _Overlapped(ctypes.Structure):
    _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]


def _kernel():
    if os.name != "nt":
        raise RuntimeError("namespace ownership requires Windows LockFileEx")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    for name, args, result in [
        ("CreateFileW", [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                         wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE], wintypes.HANDLE),
        ("LockFileEx", [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                        wintypes.DWORD, ctypes.POINTER(_Overlapped)], wintypes.BOOL),
        ("UnlockFileEx", [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                          wintypes.DWORD, ctypes.POINTER(_Overlapped)], wintypes.BOOL),
        ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
    ]:
        function = getattr(kernel, name)
        function.argtypes, function.restype = args, result
    return kernel


class _FileLock:
    def __init__(self, path: Path, exclusive: bool):
        self.kernel, self.handle, self.overlapped = _kernel(), None, _Overlapped()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.kernel.CreateFileW(str(path), 0x80000000 | 0x40000000, 3, None, 4, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = handle
        if not self.kernel.LockFileEx(handle, 1 | (2 if exclusive else 0), 0, 1, 0, ctypes.byref(self.overlapped)):
            error = ctypes.get_last_error()
            self.kernel.CloseHandle(handle)
            self.handle = None
            raise ctypes.WinError(error)

    def close(self):
        if self.handle is not None:
            if not self.kernel.UnlockFileEx(self.handle, 0, 1, 0, ctypes.byref(self.overlapped)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.kernel.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None


def _resource(identity: str):
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def verify_worker_ownership(state_root, run_id, job_name, plan, *, resources=None):
    """Read-only check of every controller-published output binding.

    This checks durable identity, not another process's file handle. The
    ManagedWorker controller must retain the native locks and the sole Job
    lifetime handle throughout inference. A worker never acquires fresh leases.
    """
    expected = NamespaceLease(state_root, run_id, job_name, plan)
    # A running worker rechecks the exact resources leased before inference;
    # newly created output inodes were not present when the lease was acquired.
    for resource, exclusive in expected.resources if resources is None else resources:
        marker = expected.state_root / "ownership" / resource / (expected.run_id + ".json")
        try:
            if marker.is_symlink():
                raise OwnershipConflict(resource)
            actual = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise OwnershipConflict(resource) from exc
        wanted = {"schema_version": "1.0", "run_id": expected.run_id,
                  "job_name": job_name, "plan_hash": plan.plan_hash,
                  "resource_id": resource, "access": "write" if exclusive else "read",
                  "state": "unsettled"}
        if actual != wanted:
            raise OwnershipConflict(resource)
    return expected.resources if resources is None else resources


class NamespaceLease:
    def __init__(self, state_root, run_id, job_name, plan):
        self.state_root, self.run_id, self.job_name = Path(state_root).resolve(), str(uuid.UUID(run_id)), job_name
        library_root = getattr(plan, "root", None)
        if library_root is not None and self.state_root.is_relative_to(Path(library_root).resolve()):
            raise ValueError("state root must be outside library")
        self.plan_hash = plan.plan_hash
        resources = {}
        for use in plan.output_ownership:
            if use.role == "source_audio":
                continue
            path = Path(use.path).resolve()
            identities = ["path:" + os.path.normcase(str(path))]
            if path.exists():
                stat = path.stat()
                identities.append(f"file:{stat.st_dev}:{stat.st_ino}")
            for identity in identities:
                key = _resource(identity)
                resources[key] = resources.get(key, False) or use.access == "write"
        self.resources = tuple(sorted(resources.items()))
        self.markers = []
        self._locks = []
        self.acquired = False
        self.settled = False

    def acquire(self):
        if self.acquired or self.settled:
            raise RuntimeError("lease cannot be acquired twice")
        try:
            for resource, exclusive in self.resources:
                directory = self.state_root / "ownership" / resource
                try:
                    lock = _FileLock(directory / "lease.lock", exclusive)
                except OSError as exc:
                    raise OwnershipConflict(resource) from exc
                self._locks.append(lock)
                for marker in directory.glob("*.json"):
                    try:
                        previous = json.loads(marker.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        raise OwnershipConflict(resource) from exc
                    if exclusive or previous.get("access") != "read":
                        raise OwnershipConflict(resource)
            for resource, exclusive in self.resources:
                marker = self.state_root / "ownership" / resource / (self.run_id + ".json")
                if marker.exists():
                    raise OwnershipConflict(resource)
                # Remember each new marker before publication, including interrupted replace.
                self.markers.append(marker)
                atomic_record(marker, {"schema_version": "1.0", "run_id": self.run_id,
                                       "job_name": self.job_name, "plan_hash": self.plan_hash,
                                       "resource_id": resource, "access": "write" if exclusive else "read",
                                       "state": "unsettled"})
            self.acquired = True
            return self
        except BaseException:
            # Nothing has launched: roll back this acquisition, never prior markers.
            for marker in self.markers:
                marker.unlink(missing_ok=True)
            self.markers.clear()
            self._release()
            raise

    def _release(self):
        while self._locks:
            self._locks[-1].close()
            self._locks.pop()

    def settle(self, cleanup_verified: bool):
        if not cleanup_verified:
            raise RuntimeError("ownership cannot settle before verified process cleanup")
        if self.settled:
            return
        for marker in self.markers:
            marker.unlink(missing_ok=True)
        self._release()
        self.acquired, self.settled = False, True

    @staticmethod
    def recover_marker(marker_path, cleanup_receipt=None):
        """Explicit recovery; never called automatically by acquire().

        An absent Job is deliberately insufficient after controller/power loss.
        Keep this blocked until prior ownership has trustworthy cleanup evidence.
        """
        marker_path = Path(marker_path)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("schema_version") != "1.0" or marker.get("state") != "unsettled":
            raise RuntimeError("unsupported ownership marker")
        resource = marker["resource_id"]
        with_lock = _FileLock(marker_path.parent / "lease.lock", True)
        try:
            verified = False
            if cleanup_receipt is not None:
                receipt = json.loads(Path(cleanup_receipt).read_text(encoding="utf-8"))
                verified = (receipt.get("schema_version") == "1.0" and receipt.get("cleanup_verified") is True
                            and receipt.get("active_count") == 0 and receipt.get("run_id") == marker["run_id"]
                            and receipt.get("plan_hash") == marker["plan_hash"]
                            and receipt.get("job_name") == marker["job_name"])
            if not verified:
                runtime_name = "scripts.asmr_gui_runtime" if __package__.startswith("scripts.") else "asmr_gui_runtime"
                WindowsJob = importlib.import_module(runtime_name).WindowsJob
                try:
                    job = WindowsJob(marker["job_name"], create=False)
                except OSError as exc:
                    raise OwnershipConflict(resource) from exc
                try:
                    if job.active_count() != 0:
                        raise OwnershipConflict(resource)
                    state_root = marker_path.resolve().parents[2]
                    run_dir = state_root / "runs" / marker["run_id"]
                    recovered = reconcile_after_cleanup(run_dir, marker["run_id"], marker["plan_hash"])
                    atomic_record(run_dir / "recovery.json", recovered)
                    atomic_record(run_dir / "cleanup.json", {
                        "schema_version": "1.0", "run_id": marker["run_id"],
                        "plan_hash": marker["plan_hash"], "job_name": marker["job_name"],
                        "cleanup_verified": True, "active_count": 0,
                        "state": "stopped", "source": "explicit_empty_job_recovery",
                        "reconciliation": recovered})
                finally:
                    job.close()
            marker_path.unlink()
        finally:
            with_lock.close()
