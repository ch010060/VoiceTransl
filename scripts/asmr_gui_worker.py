"""Isolated, contained ASMR worker. stdout is exclusively protocol v1 JSONL."""
from __future__ import annotations

import argparse
import ctypes
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
from uuid import UUID

if __package__ == "scripts":
    from .asmr_gui_runtime import assert_current_in_job
    from .asmr_transcribe_lib.gui_plan import build_execution_plan
    from .asmr_transcribe_lib.gui_plan_io import load_plan, save_plan
    from .asmr_transcribe_lib.gui_preflight import preflight
    from .asmr_transcribe_lib.gui_ownership import verify_worker_ownership
    from .asmr_transcribe_lib.gui_protocol import MAX_LINE, ProtocolError, validate_event
    from .asmr_transcribe_lib.gui_results import RunStore, atomic_json, read_json
    from .asmr_transcribe_lib.runtime import CancellationToken
else:
    from asmr_gui_runtime import assert_current_in_job
    from asmr_transcribe_lib.gui_plan import build_execution_plan
    from asmr_transcribe_lib.gui_plan_io import load_plan, save_plan
    from asmr_transcribe_lib.gui_preflight import preflight
    from asmr_transcribe_lib.gui_ownership import verify_worker_ownership
    from asmr_transcribe_lib.gui_protocol import MAX_LINE, ProtocolError, validate_event
    from asmr_transcribe_lib.gui_results import RunStore, atomic_json, read_json
    from asmr_transcribe_lib.runtime import CancellationToken


def default_state_root():
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise ValueError("LOCALAPPDATA is unavailable")
    return Path(base) / "VoiceTransl" / "asmr-batch"


@contextmanager
def protocol_streams():
    """Private native/Python output cannot corrupt JSONL or disclose captions.

    The protocol writer uses a duplicate of original fd1. Redirecting fd1/2
    also covers inherited native descendants, not only Python print calls.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_out, saved_err = os.dup(1), os.dup(2)
    except OSError:
        raise ProtocolError("protocol streams unavailable") from None
    stream = os.fdopen(os.dup(saved_out), "w", encoding="utf-8", newline="\n", buffering=1)
    diagnostic = os.fdopen(os.dup(saved_err), "w", encoding="utf-8", newline="\n", buffering=1)
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            # CRT fd redirection alone does not define Win32 inherited handles.
            # Native processes which use GetStdHandle must see the same sink.
            import msvcrt
            from ctypes import wintypes
            set_standard = ctypes.WinDLL("kernel32", use_last_error=True).SetStdHandle
            set_standard.argtypes = [wintypes.DWORD, wintypes.HANDLE]
            set_standard.restype = wintypes.BOOL
            for number, descriptor in ((-11, 1), (-12, 2)):
                if not set_standard(number, msvcrt.get_osfhandle(descriptor)):
                    raise ctypes.WinError(ctypes.get_last_error())
            with redirect_stdout(sink), redirect_stderr(sink):
                yield stream, diagnostic
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        if "set_standard" in locals():
            for number, descriptor in ((-11, 1), (-12, 2)):
                set_standard(number, msvcrt.get_osfhandle(descriptor))
        os.close(saved_out)
        os.close(saved_err)
        stream.close()
        diagnostic.close()


def _adapters(plan, native_metadata):
    # Imports are local so scan/preflight never instantiate backend models.
    if __package__ == "scripts":
        from .asmr_transcribe_lib.kotoba_faster_whisper import run_kotoba_faster_whisper_transcription
        from .asmr_transcribe_lib.translation import translate_srt_to_simplified
        from .asmr_transcribe_lib.opencc_utils import convert_srt_file_opencc
        from .asmr_transcribe_lib.gui_native_adapter import strict_xxl_transcriber
    else:
        from asmr_transcribe_lib.kotoba_faster_whisper import run_kotoba_faster_whisper_transcription
        from asmr_transcribe_lib.translation import translate_srt_to_simplified
        from asmr_transcribe_lib.opencc_utils import convert_srt_file_opencc
        from asmr_transcribe_lib.gui_native_adapter import strict_xxl_transcriber
    settings = plan.settings
    def kotoba_local(**kwargs):
        from faster_whisper import WhisperModel
        def local_model(*args, **options):
            return WhisperModel(*args, local_files_only=True, **options)
        return run_kotoba_faster_whisper_transcription(**kwargs, whisper_model_cls=local_model)
    def translate(source, target):
        return translate_srt_to_simplified(source, target, base_url=settings["translate_api"],
            model=settings["translate_model"], max_blocks=settings["translate_chunk_size"])
    return {"transcriber": kotoba_local if settings["transcribe_backend"] == "kotoba" else strict_xxl_transcriber(plan, native_metadata),
            "translator": translate if plan.mode == "full" else None,
            "opencc_converter": convert_srt_file_opencc}


def _helper_ready(stream, run_id):
    event = {"schema_version": "1.0", "event": "worker_ready", "run_id": run_id, "seq": 1,
             "emitted_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
             "elapsed_ms": 0, "work_id": None, "track_id": None, "track_revision": None,
             "stage": None, "status": "ready", "payload": {"protocol": "1.0", "containment_ready": True}}
    validate_event(event, run_id)
    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    stream.flush()


def _run(plan, args, run_dir, stream, adapters, failure_context):
    verify_worker_ownership(args.state_root or default_state_root(), args.run_id, args.job_name, plan)
    native_metadata = {}
    store = RunStore(run_dir, args.run_id, plan, stream, asr_metadata=native_metadata)
    failure_context["store"] = store
    store.emit("worker_ready", status="ready", payload={"protocol": "1.0", "containment_ready": True})
    check = adapters.get("preflight", preflight) if adapters is not None else preflight
    availability = check(plan, Path(args.project_root))
    atomic_json(run_dir / "preflight.json", availability)
    if not availability.get("available"):
        return _failure(run_dir, args.run_id, plan.plan_hash, "preflight_failed", 3, "PREFLIGHT_FAILED", store)
    plan.assert_ready()
    if __package__ == "scripts":
        from .asmr_transcribe_lib.gui_execution import run_plan
    else:
        from asmr_transcribe_lib.gui_execution import run_plan
    token, done = CancellationToken(), threading.Event()
    failures = []

    def controls():
        try:
            # No buffered stdin reader may survive interpreter shutdown while
            # holding its Python buffer lock. Peek the native pipe, then read
            # only available bytes; this thread always observes done promptly.
            import msvcrt
            from ctypes import wintypes
            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
            peek.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                             ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            peek.restype = wintypes.BOOL
            pending = bytearray()
            while not done.is_set():
                available = wintypes.DWORD()
                if not peek(handle, None, 0, None, ctypes.byref(available), None):
                    token.cancel()
                    return
                if not available.value:
                    done.wait(.02)
                    continue
                pending.extend(os.read(sys.stdin.fileno(), min(available.value, MAX_LINE + 1)))
                while b"\n" in pending:
                    line, _, tail = pending.partition(b"\n")
                    pending = bytearray(tail)
                    if len(line) > MAX_LINE:
                        raise ProtocolError("invalid control framing")
                    message = json.loads(line.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                    store.ack_control(message, token)
                if len(pending) > MAX_LINE:
                    raise ProtocolError("invalid control framing")
        except Exception:
            if not done.is_set():
                failures.append("CONTROL_PROTOCOL_FAILED")
                token.cancel()

    def heartbeat():
        try:
            while not done.wait(1):
                snapshot = store.snapshot()
                active = sum(track.get("status") == "running" for track in snapshot["tracks"].values())
                store.emit("heartbeat", payload={"last_seq": snapshot["last_seq"], "active_count": active})
        except Exception:
            if not done.is_set():
                failures.append("PROTOCOL_DELIVERY_FAILED")
                token.cancel()

    store.start()
    threads = [threading.Thread(target=controls, daemon=True), threading.Thread(target=heartbeat, daemon=True)]
    for thread in threads:
        thread.start()
    try:
        selected = _adapters(plan, native_metadata) if adapters is None else {key: value for key, value in adapters.items() if key != "preflight"}
        results = run_plan(plan, project_root=Path(args.project_root), observer=store.observer,
                           cancel_token=token, **selected)
    finally:
        done.set()
        for thread in threads:
            thread.join(timeout=1)
    if failures:
        return _failure(run_dir, args.run_id, plan.plan_hash, "protocol_failed", 4, failures[0], store)
    return store.finish(results)["exit_code"]


def _failure(run_dir, run_id, plan_hash, outcome, code, error_code, store=None):
    snapshot = store.snapshot() if store else (read_json(run_dir / "state.json") if (run_dir / "state.json").exists() else {})
    atomic_json(run_dir / "result.json", {"schema_version": "1.0", "run_id": run_id,
        "plan_hash": plan_hash, "outcome": outcome, "exit_code": code, "complete": False,
        "counts": snapshot.get("counts", {}), "error": {"code": error_code}})
    return code


def _reserve_helper_dir(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise ValueError("helper run directory is not new")


def main(argv=None, *, adapters=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["scan", "preflight", "run"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--state-root")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--plan")
    parser.add_argument("--expected-digest")
    parser.add_argument("--root")
    parser.add_argument("--mode", choices=["full", "transcribe_only"], default="full")
    parser.add_argument("--settings-file")
    parser.add_argument("--skip-existing-subtitles", action="store_true",
                        help="GUI policy: skip existing common subtitles without conversion")
    args = parser.parse_args(argv)
    with protocol_streams() as (stream, diagnostic):
        run_dir, plan = None, None
        failure_context = {}
        try:
            args.run_id = str(UUID(args.run_id))
            # Every entry point is contained, including operation-scoped helpers.
            assert_current_in_job(args.job_name)
            state_root = Path(args.state_root).resolve() if args.state_root else default_state_root().resolve()
            if args.operation == "scan":
                if not args.root:
                    raise ValueError("scan requires root")
                root = Path(args.root).resolve()
                if state_root.is_relative_to(root):
                    raise ValueError("state root must be outside library")
                settings = json.loads(Path(args.settings_file).read_text(encoding="utf-8")) if args.settings_file else None
                plan = build_execution_plan(root, mode=args.mode, settings=settings,
                                            skip_existing_subtitles=args.skip_existing_subtitles)
                helper_dir = state_root / "runs" / args.run_id
                _reserve_helper_dir(helper_dir)
                run_dir = helper_dir
                failure_context["helper"] = True
                digest = save_plan(plan, run_dir / "plan.json")
                atomic_json(run_dir / "scan.json", {"schema_version": "1.0", "run_id": args.run_id,
                    "operation": "scan", "plan_hash": plan.plan_hash, "transport_hash": digest,
                    "executable": plan.executable, "work_ids": [w.work_id for w in plan.works],
                    "track_ids": [t.track_id for t in plan.planned_tracks],
                    "issue_codes": [issue.code for issue in plan.issues]})
                _helper_ready(stream, args.run_id)
                return 0 if plan.executable else 3
            if not args.plan or not args.expected_digest:
                raise ValueError("operation requires frozen plan and independent digest")
            plan = load_plan(args.plan, args.expected_digest)
            if state_root.is_relative_to(plan.root.resolve()):
                raise ValueError("state root must be outside library")
            plan.assert_ready()
            run_dir = state_root / "runs" / args.run_id
            if args.operation == "preflight":
                _reserve_helper_dir(run_dir)
                failure_context["helper"] = True
                availability = preflight(plan, Path(args.project_root))
                atomic_json(run_dir / "preflight.json", {**availability, "run_id": args.run_id, "operation": "preflight"})
                _helper_ready(stream, args.run_id)
                return 0 if availability["available"] else 3
            # The controller reserves the run directory and serializes this exact
            # plan there after acquiring namespace ownership, before bootstrap.
            if Path(args.plan).resolve() != run_dir / "plan.json":
                raise ValueError("run requires its own frozen plan")
            return _run(plan, args, run_dir, stream, adapters, failure_context)
        except Exception:
            diagnostic.write("ASMR_WORKER_OPERATION_FAILED\n")
            diagnostic.flush()
            if failure_context and run_dir is not None and run_dir.exists() and not (run_dir / "result.json").exists():
                _failure(run_dir, args.run_id, plan.plan_hash if plan else None,
                         "protocol_failed", 4, "WORKER_OPERATION_FAILED", failure_context.get("store"))
            return 4


if __name__ == "__main__":
    raise SystemExit(main())
