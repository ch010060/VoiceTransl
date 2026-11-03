"""Controlled subprocesses for native Job tests; never finds or kills by name."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def publish(directory, name, **payload):
    target = Path(directory) / (name + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), **payload}), encoding="utf-8")
    temporary.replace(target)


def wait_file(path, timeout=15):
    deadline = time.monotonic() + timeout
    while not Path(path).exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(0.02)


def tree(args):
    from scripts.asmr_gui_runtime import assert_current_in_job
    assert_current_in_job(args.job_name)
    child = None
    if args.role != "grandchild":
        next_role = "child" if args.role == "worker" else "grandchild"
        child = subprocess.Popen([
            sys.executable, "-B", "-u", __file__, "tree", "--directory", args.directory,
            "--job-name", args.job_name, "--mode", args.mode, "--role", next_role,
        ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    publish(args.directory, args.role, parent_pid=os.getppid(), child_pid=child.pid if child else None,
            containment_verified=True, job_name=args.job_name)
    stop_file = Path(args.directory) / "cooperative.stop"
    if args.role == "worker":
        wait_file(Path(args.directory) / "grandchild.json")
        publish(args.directory, "tree_ready")
        if args.mode == "early_exit":
            return 0
        def controls():
            for line in sys.stdin:
                message = json.loads(line)
                if message.get("command") == "cancel":
                    publish(args.directory, "cancel_received", request_id=message.get("request_id"))
                    if args.mode == "cooperative":
                        stop_file.touch()
        threading.Thread(target=controls, daemon=True).start()
    while not stop_file.exists():
        time.sleep(0.02)
    if child:
        child.wait(timeout=10)
    publish(args.directory, args.role + "_exited")
    return 0


def controller(args):
    from scripts.asmr_gui_runtime import ManagedWorker
    from scripts.asmr_transcribe_lib.gui_plan import build_execution_plan
    worker = ManagedWorker(build_execution_plan(args.library, mode="transcribe_only"), args.state_root,
        [sys.executable, "-B", "-u", __file__, "tree", "--directory", args.directory,
         "--job-name", "{job_name}", "--mode", "forced"],
        run_id=args.run_id, cooperative_timeout=0.25, cleanup_timeout=3)
    worker.start()
    wait_file(Path(args.directory) / "tree_ready.json")
    publish(args.directory, "controller_ready", run_id=worker.run_id, job_name=worker.job.name,
            root_pid=worker.process.pid, markers=[str(p) for p in worker.lease.markers],
            active_count=worker.job.active_count(), owned_process_ids=worker.job.process_ids())
    for line in sys.stdin:
        if line.strip() == "close":
            publish(args.directory, "controller_closed", result=worker.close())
            return 0
        if line.strip() == "crash":
            os._exit(73)
    # EOF deliberately behaves like a normal controller close.
    worker.close()
    return 0


def assignment_failure(args):
    import ctypes
    from scripts import asmr_gui_runtime as runtime
    def fail_native_assignment(self):
        # Actual native assignment with an invalid Job HANDLE; child execution must be gated.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel32.AssignProcessToJobObject.restype = ctypes.c_int
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        assert not kernel32.AssignProcessToJobObject(None, kernel32.GetCurrentProcess())
        raise ctypes.WinError(ctypes.get_last_error())
    runtime.WindowsJob.assign_current = fail_native_assignment
    return runtime.bootstrap(args.job_name, [
        sys.executable, "-B", "-u", __file__, "sentinel", "--directory", args.directory])


def server(args):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"data":[{"id":"synthetic-sakura"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    instance = HTTPServer(("127.0.0.1", 0), Handler)
    publish(args.directory, "server", port=instance.server_port)
    instance.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--job-name")
    parser.add_argument("--role", default="worker")
    parser.add_argument("--mode", default="forced")
    parser.add_argument("--library")
    parser.add_argument("--state-root")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.operation == "tree":
        return tree(args)
    if args.operation == "controller":
        return controller(args)
    if args.operation == "assignment_failure":
        return assignment_failure(args)
    if args.operation == "server":
        return server(args)
    if args.operation == "sentinel":
        publish(args.directory, "inference_started")
        return 0
    if args.operation == "dummy":
        publish(args.directory, "dummy")
        time.sleep(120)
        return 0
    parser.error("unknown operation")


if __name__ == "__main__":
    raise SystemExit(main())
