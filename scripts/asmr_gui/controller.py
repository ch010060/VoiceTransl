"""GUI-thread state and a single background lifecycle owner, not a scheduler."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from uuid import uuid4

from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal

from scripts.asmr_gui_runtime import ManagedOperation, ManagedWorker
from scripts.asmr_gui_worker import default_state_root
from scripts.asmr_transcribe_lib.gui_plan import PlanRejected, build_execution_plan, file_stamp
from scripts.asmr_transcribe_lib.gui_plan_io import encode_plan, load_plan, save_plan
from scripts.asmr_transcribe_lib.gui_protocol import JSONLDecoder, ProtocolError
from scripts.asmr_transcribe_lib.gui_results import (
    atomic_json, read_json, reconcile_run, resume_provenance, safe_ref,
)
from .run_model import RunModel
from .diagnostics import DiagnosticLog, plan_rejection_diagnostic
from .health import HealthMonitor

BUSY = {"SCANNING", "PREFLIGHT", "STARTING", "RUNNING", "STOPPING", "RECONCILING", "CLEANUP_FAILED"}


@dataclass(frozen=True)
class Intent:
    generation: int
    operation_id: str
    kind: str
    root: str
    settings_json: str
    mode: str
    limit: int | None
    selected: tuple[str, ...]
    catalog: object
    resume_runs: tuple
    stop: threading.Event


class Controller(QObject):
    state_changed = pyqtSignal(str)
    catalog_changed = pyqtSignal()
    backend_changed = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    completed = pyqtSignal(dict)
    close_ready = pyqtSignal()
    settings_changed = pyqtSignal()
    _delivery = pyqtSignal(object)

    def __init__(self, *, state_root=None, project_root=None, python_executable=None,
                 operation_factory=ManagedOperation, worker_factory=ManagedWorker,
                 worker_command=None, health_monitor_factory=None, parent=None):
        super().__init__(parent)
        self.run_model = RunModel(self)
        self.state = "IDLE"
        self.root, self.settings, self.mode, self.limit = "", {}, "full", None
        self.state_root = Path(state_root) if state_root is not None else default_state_root()
        self.project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
        self.python_executable = str(python_executable or sys.executable)
        self.operation_factory, self.worker_factory = operation_factory, worker_factory
        self.worker_command = worker_command
        self.last_run_dir, self.last_backend, self.last_error = None, {}, ""
        self.last_cleanup = None
        self._generation, self._op_id = 0, None
        self._intent, self._owned = None, None
        self._owned_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asmr-controller")
        self._future, self._closing, self._retrying = None, False, False
        self._completed_ids = set()
        self._trusted_runs = []
        self._history_file = self.state_root / "gui-last-result.json"
        self.diagnostic_log_path = self.state_root / "logs" / ("gui-" + str(uuid4()) + ".log")
        self._diagnostics = DiagnosticLog(self.diagnostic_log_path)
        self.log_message.connect(self._record_diagnostic)
        self._events = queue.Queue(maxsize=2048)
        self._delivery.connect(self._receive, Qt.ConnectionType.QueuedConnection)
        self._drain_timer = QTimer(self)
        self._drain_timer.setInterval(33)
        self._drain_timer.timeout.connect(self._drain_events)
        self._drain_timer.start()
        self.run_model.selection_changed.connect(self._selection_changed)
        self._health = (health_monitor_factory or HealthMonitor)(project_root=self.project_root, parent=self)
        self._health.updated.connect(self._health_updated)
        self._health.configure(self.settings, self.mode)

    @property
    def busy(self):
        return self.state in BUSY

    @property
    def safe_to_edit(self):
        return not self.busy and not self._closing

    def _set_state(self, state):
        if self.state != state:
            self.state = state
            self.run_model.lifecycle = state
            self.run_model.locked = state in BUSY
            self._health.set_paused(state in BUSY)
            self.state_changed.emit(state)
            self.log_message.emit("lifecycle · " + state)

    def _record_diagnostic(self, message):
        self._diagnostics.append(self.root, message, f"generation={self._generation} operation={self._op_id or '-'}")

    def _invalidate(self):
        self._generation += 1

    def _selection_changed(self):
        if self.safe_to_edit:
            self._invalidate()

    def set_root(self, root):
        if not self.safe_to_edit:
            return False
        root = str(root).strip()
        if self.root != root:
            self.root = root
            self._invalidate()
            self.run_model.load_plan(None)
            self._set_state("IDLE")
            self.catalog_changed.emit()
        return True

    def apply_settings(self, settings, mode="full", limit=None):
        if not self.safe_to_edit or mode not in {"full", "transcribe_only"}:
            return False
        if limit is not None and (type(limit) is not int or limit < 1):
            return False
        # JSON copy prevents a settings dialog draft retaining mutable ownership.
        copied = json.loads(json.dumps(settings, ensure_ascii=False))
        self.settings, self.mode, self.limit = copied, mode, limit
        self._invalidate()
        self.last_backend = {}
        self.backend_changed.emit({})
        self._health.configure(self.settings, self.mode)
        self.settings_changed.emit()
        return True

    def clear(self):
        if not self.safe_to_edit:
            return False
        self._invalidate()
        self.run_model.load_plan(None)
        self.last_error = ""
        self._set_state("IDLE")
        self.catalog_changed.emit()
        return True

    def _new_intent(self, kind):
        self._op_id = str(uuid4())
        catalog = self.run_model.plan
        selected = tuple(work.work_id for work in catalog.works if work.work_id in self.run_model.selected_work_ids) if catalog else ()
        self._intent = Intent(self._generation, self._op_id, kind, self.root,
            json.dumps(self.settings, sort_keys=True, ensure_ascii=False), self.mode,
            self.limit, selected, catalog, tuple(self._trusted_runs), threading.Event())
        return self._intent

    def scan(self, root=None):
        if not self.safe_to_edit:
            return False
        if root is not None and not self.set_root(root):
            return False
        if not self.root:
            return False
        intent = self._new_intent("scan")
        self.last_error = ""
        self._set_state("SCANNING")
        self._future = self._executor.submit(self._scan, intent)
        return True

    def start(self):
        if not self.safe_to_edit or self.run_model.plan is None or not self.run_model.selected_work_ids:
            return False
        intent = self._new_intent("run")
        self.last_error = ""
        self._set_state("PREFLIGHT")
        self._future = self._executor.submit(self._execute, intent)
        return True

    def check_backends(self):
        if not self.safe_to_edit:
            return False
        self.log_message.emit("後端可用性檢查中（不執行推論）")
        return self._health.check()

    def _health_updated(self, result):
        if self._closing or self.busy:
            return
        self.last_backend = result
        self.backend_changed.emit(result)

    def stop(self):
        if not self.busy or self._intent is None:
            return False
        self._intent.stop.set()
        if self.state == "CLEANUP_FAILED":
            if self._retrying:
                return True
            with self._owned_lock:
                managed = self._owned
            if managed is None:
                return False
            self._retrying = True
            self._set_state("STOPPING")
            self._future = self._executor.submit(self._retry_cleanup, self._intent, managed)
        else:
            self._set_state("STOPPING")
        return True

    def request_close(self):
        self._health.close()
        if self.busy:
            self._closing = True
            self.stop()
            return False
        self._closing = True
        self._drain_timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._diagnostics.close()
        return True

    def _post(self, intent, kind, value):
        self._delivery.emit((intent.generation, intent.operation_id, kind, value))

    def _command(self, managed, operation, extras=()):
        return [self.python_executable, "-B", "-u", str(self.project_root / "scripts" / "asmr_gui_worker.py"),
            operation, "--run-id", managed.run_id, "--job-name", managed.job.name,
            "--state-root", str(self.state_root), "--project-root", str(self.project_root), *map(str, extras)]

    def _drive(self, intent, managed, on_tick=None):
        if intent.stop.is_set():
            return {"cleanup_verified": True, "state": "stopped", "result": {"outcome": "operation_cancelled", "complete": False}}
        with self._owned_lock:
            self._owned = managed
        try:
            managed.start()
            if intent.kind == "run" and isinstance(managed, ManagedWorker) and not isinstance(managed, ManagedOperation):
                self._post(intent, "phase", "RUNNING")
            while True:
                if intent.stop.is_set():
                    result = managed.stop()
                    break
                try:
                    # Bounded waits only in the lifecycle executor; Qt remains free.
                    result = managed.wait(timeout=.15)
                    break
                except subprocess.TimeoutExpired:
                    if on_tick is not None:
                        on_tick()
                    continue
            if result.get("cleanup_verified"):
                with self._owned_lock:
                    if self._owned is managed:
                        self._owned = None
            return result
        except Exception:
            # A start failure with a live process must still follow native cleanup.
            if getattr(managed, "process", None) is not None:
                result = managed.stop()
                if not result.get("cleanup_verified"):
                    return result
            with self._owned_lock:
                if self._owned is managed:
                    self._owned = None
            raise

    def _draft_file(self, intent):
        root = Path(intent.root).resolve()
        if self.state_root.resolve().is_relative_to(root):
            raise ValueError("state root must be outside library")
        path = self.state_root / "drafts" / (intent.operation_id + ".json")
        atomic_json(path, json.loads(intent.settings_json))
        return path

    def _scan(self, intent):
        try:
            settings = self._draft_file(intent)
            managed = self.operation_factory("scan", self.state_root,
                lambda operation: self._command(operation, "scan", ["--root", intent.root,
                    "--mode", intent.mode, "--settings-file", settings, "--skip-existing-subtitles"]), library_root=intent.root)
            cleanup = self._drive(intent, managed)
            if not cleanup.get("cleanup_verified"):
                self._post(intent, "cleanup_failed", cleanup)
                return
            if intent.stop.is_set():
                self._post(intent, "cancelled", cleanup)
                return
            if cleanup.get("result", {}).get("complete") is not True:
                raise ProtocolError("scan operation incomplete")
            info = read_json(managed.run_dir / "scan.json")
            plan = load_plan(managed.run_dir / "plan.json", info["transport_hash"])
            proofs = self._provenance(intent)
            if proofs:
                # Readonly worker discovery remains the catalog source. An
                # explicit proof-aware plan uses only previously verified runs,
                # never promotion based on surviving SRTs alone.
                verified = build_execution_plan(intent.root, mode=intent.mode,
                    settings=json.loads(intent.settings_json), provenance=proofs, skip_existing_subtitles=True)
                if plan.duplicate_groups != verified.duplicate_groups or [t.track_id for t in plan.tracks] != [t.track_id for t in verified.tracks]:
                    raise ValueError("catalog changed during provenance resolution")
                plan = verified
            self._post(intent, "catalog", (plan, managed.run_dir, self._historical_display(intent, plan)))
        except Exception as exc:
            self._error(intent, "SCAN_FAILED", exc)

    def _historical_display(self, intent, current_plan):
        """Restore trusted *presentation only*, including pre-index GUI sessions.

        Never grant resume permissions from discovered disk artifacts.
        """
        if any(Path(root).resolve() == Path(intent.root).resolve()
               for _, _, _, root in intent.resume_runs):
            # An explicit rescan in this session retains the existing READY /
            # receipt-aware resume workflow, rather than a historical overlay.
            return None
        run_root = (self.state_root / "runs").resolve()
        candidates = []
        try:
            pointer = read_json(self._history_file)
            if pointer.get("schema") == 1 and Path(pointer.get("root", "")).resolve() == Path(intent.root).resolve():
                candidates.append(Path(pointer["directory"]))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        try:
            # Flat, bounded history discovery; no recursive data-library scan.
            recent = sorted((p for p in run_root.iterdir() if p.is_dir() and
                             (p / "result.json").is_file()),
                            key=lambda p: p.stat().st_mtime_ns, reverse=True)[:128]
            candidates.extend(recent)
        except OSError:
            pass
        seen = set()
        for candidate in candidates:
            try:
                directory = candidate.resolve()
                if directory in seen or directory.parent != run_root:
                    continue
                seen.add(directory)
                old_plan = load_plan(directory / "plan.json")
                if Path(old_plan.root).resolve() != Path(intent.root).resolve():
                    continue
                current = {t.track_id: tuple(t.source_stamp) for t in current_plan.tracks}
                original = {t.track_id: tuple(t.source_stamp) for t in old_plan.tracks}
                if not current or current != original:
                    continue
                # A changed matching policy is not an equivalent prior catalog.
                if old_plan.duplicate_groups != current_plan.duplicate_groups:
                    continue
                historical_id = read_json(directory / "result.json").get("run_id")
                if not isinstance(historical_id, str):
                    continue
                verified = reconcile_run(directory, historical_id, old_plan.plan_hash)
                if (verified.get("complete") and verified.get("outcome") == "completed"
                        and set(verified.get("tracks", {})) == set(current)):
                    return verified, directory
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        return None

    def _save_historical_pointer(self, result, directory):
        if result.get("complete") and result.get("outcome") == "completed":
            try:
                atomic_json(self._history_file, {"schema": 1, "root": self.root,
                    "directory": str(Path(directory).resolve()), "run_id": result["run_id"]})
            except (OSError, KeyError, ValueError) as exc:
                self.log_message.emit("歷史顯示索引保存失敗 · " + type(exc).__name__)

    def _provenance(self, intent):
        proofs = []
        for directory, run_id, plan_hash, root in intent.resume_runs:
            if Path(root).resolve() != Path(intent.root).resolve():
                continue
            try:
                proofs.extend(resume_provenance(directory, run_id, plan_hash))
            except (OSError, ValueError):
                # Invalid/changed prior artifacts confer no trust. Planning
                # retains its explicit UNVERIFIED_LEGACY block for those files.
                continue
        return tuple(proofs)

    def _freeze(self, intent):
        if intent.catalog is None:
            raise ValueError("catalog is unavailable")
        # Validate the displayed scan before an explicit new selection/config plan.
        # The new plan is rejected if catalog/group membership changed underneath.
        if any(file_stamp(path) != stamp for path, stamp in intent.catalog.inventory):
            raise ValueError("catalog inventory changed; rescan required")
        try:
            encode_plan(intent.catalog)
        except PlanRejected as exc:
            exc.diagnostic_origin = "catalog_freshness_validation"
            raise
        if intent.stop.is_set():
            return None
        try:
            plan = build_execution_plan(intent.root, mode=intent.mode,
                selected_work_ids=intent.selected, limit=intent.limit,
                settings=json.loads(intent.settings_json), provenance=self._provenance(intent), skip_existing_subtitles=True)
        except PlanRejected as exc:
            exc.diagnostic_origin = "plan_construction"
            raise
        old = [(track.track_id, track.source_stamp) for track in intent.catalog.tracks]
        if old != [(track.track_id, track.source_stamp) for track in plan.tracks] or intent.catalog.duplicate_groups != plan.duplicate_groups:
            raise ValueError("catalog changed; rescan required")
        try:
            plan.assert_ready()
        except PlanRejected as exc:
            exc.diagnostic_origin = "plan_construction" if plan.issues else "start_freshness_validation"
            raise
        return plan

    def _preflight(self, intent, plan):
        directory = self.state_root / "plans" / intent.operation_id
        digest = save_plan(plan, directory / "plan.json")
        managed = self.operation_factory("preflight", self.state_root,
            lambda operation: self._command(operation, "preflight", ["--plan", directory / "plan.json",
                "--expected-digest", digest]), library_root=plan.root, expected_plan_hash=plan.plan_hash)
        cleanup = self._drive(intent, managed)
        if not cleanup.get("cleanup_verified"):
            self._post(intent, "cleanup_failed", cleanup)
            return None
        if intent.stop.is_set():
            self._post(intent, "cancelled", cleanup)
            return None
        if cleanup.get("result", {}).get("complete") is not True:
            raise ProtocolError("preflight operation incomplete")
        checks = read_json(managed.run_dir / "preflight.json")
        self._post(intent, "backend", checks)
        return checks

    def _execute(self, intent):
        try:
            plan = self._freeze(intent)
            if plan is None:
                self._post(intent, "cancelled", {})
                return
            checks = self._preflight(intent, plan)
            if checks is None:
                return
            if checks.get("available") is not True:
                self._post(intent, "error", "PREFLIGHT_FAILED")
                return
            if intent.stop.is_set():
                self._post(intent, "cancelled", {})
                return
            self._post(intent, "phase", "STARTING")
            decoder_lock, lost = threading.Lock(), threading.Event()
            decoder = None

            def notify(data):
                with decoder_lock:
                    if lost.is_set():
                        return
                    try:
                        for event in decoder.feed(data):
                            self._events.put_nowait((intent.generation, intent.operation_id, event))
                    except (ProtocolError, queue.Full):
                        lost.set()
                        self._post(intent, "delivery_gap", True)

            def command(managed):
                nonlocal decoder
                digest = save_plan(plan, managed.run_dir / "plan.json")
                decoder = JSONLDecoder(managed.run_id)
                if self.worker_command:
                    return self.worker_command(managed)
                return self._command(managed, "run", ["--plan", managed.run_dir / "plan.json", "--expected-digest", digest])

            managed = self.worker_factory(plan, self.state_root, command_factory=command, on_stdout=notify)
            self._post(intent, "begin", (plan, managed.run_id, managed.run_dir))
            last_snapshot = 0
            def recover_live():
                nonlocal last_snapshot
                now = time.monotonic()
                if not lost.is_set() or now - last_snapshot < .5:
                    return
                last_snapshot = now
                try:
                    recovered = reconcile_run(managed.run_dir, managed.run_id, plan.plan_hash)
                    snapshot = read_json(managed.run_dir / "state.json")
                    if (snapshot.get("run_id") != managed.run_id or snapshot.get("plan_hash") != plan.plan_hash
                            or snapshot.get("schema_version") != "1.0"
                            or set(snapshot.get("tracks", {})) != {t.track_id for t in plan.planned_tracks}):
                        raise ProtocolError("snapshot identity differs from frozen run")
                    # Snapshot alone cannot certify terminal outcomes. Merge only
                    # receipts already verified by the existing reconciler.
                    terminal = {"success", "skipped", "copied", "transcribed", "failed", "cancelled", "not_started"}
                    tracks = {key: value for key, value in snapshot["tracks"].items() if value.get("status") not in terminal}
                    tracks.update(recovered["tracks"])
                    snapshot["tracks"] = tracks
                    self._post(intent, "snapshot", snapshot)
                except (OSError, ValueError, KeyError):
                    # Atomic state/receipts may not exist before worker_ready.
                    # Preserve the gap and retry a bounded check on a later tick.
                    return
            cleanup = self._drive(intent, managed, recover_live)
            if not cleanup.get("cleanup_verified"):
                self._post(intent, "cleanup_failed", cleanup)
                return
            self._post(intent, "phase", "RECONCILING")
            if getattr(managed, "process", None) is None:
                self._post(intent, "cancelled", cleanup)
                return
            with decoder_lock:
                if decoder is not None and not lost.is_set():
                    try:
                        decoder.finish()
                    except ProtocolError:
                        lost.set()
            # Final authority is independent of delivered stdout. ManagedWorker
            # has already verified Job0/root exit and reconciled before settlement.
            result = self._enrich(cleanup.get("result", {}), managed.run_dir)
            self._post(intent, "finished", (cleanup, result, managed.run_dir))
        except Exception as exc:
            self._error(intent, "RUN_FAILED", exc)

    def _enrich(self, result, run_dir):
        result = json.loads(json.dumps(result))
        timings = {}
        decoder = JSONLDecoder(result["run_id"]) if result.get("run_id") else None
        if decoder is not None:
            try:
                with (run_dir / "events.jsonl").open("rb") as journal:
                    while chunk := journal.read(65536):
                        for event in decoder.feed(chunk):
                            key = event.get("track_id")
                            if event["event"] == "track_started":
                                timings.setdefault(key, {})["started_ms"] = event["elapsed_ms"]
                            elif event["event"] in {"track_finished", "track_failed", "track_cancelled", "track_not_started"}:
                                timing = timings.setdefault(key, {})
                                timing["ended_ms"] = event["elapsed_ms"]
                                if "reason" in event["payload"]:
                                    timing["reason"] = event["payload"]["reason"]
                                elif event["event"] == "track_failed":
                                    timing["reason"] = event["payload"].get("error", {}).get("code", "TRACK_EXECUTION_FAILED")
                decoder.finish()
            except (OSError, ProtocolError):
                # Only timings read from valid journal frames are retained.
                pass
        for track_id, track in result.get("tracks", {}).items():
            track.update(timings.get(track_id, {}))
            ref = track.get("receipt_ref")
            if ref:
                receipt = read_json(safe_ref(run_dir, ref))
                track["warnings"] = receipt.get("warnings", [])
        return result

    def _retry_cleanup(self, intent, managed):
        try:
            cleanup = managed.stop()
            if not cleanup.get("cleanup_verified"):
                self._post(intent, "cleanup_failed", cleanup)
                return
            with self._owned_lock:
                self._owned = None
            if intent.kind == "run" and getattr(managed, "plan", None) is not None:
                result = self._enrich(cleanup.get("result", {}), managed.run_dir)
                self._post(intent, "finished", (cleanup, result, managed.run_dir))
            else:
                self._post(intent, "cancelled", cleanup)
        except Exception as exc:
            self._error(intent, "CLEANUP_RETRY_FAILED", exc)

    def _error(self, intent, code, exc):
        # Fixed diagnostic categories only, never raw exceptions/transcriptions.
        with self._owned_lock:
            retained = self._owned is not None
        if retained:
            self._post(intent, "cleanup_failed", {"cleanup_verified": False, "error": code})
        elif isinstance(exc, PlanRejected):
            report, summary = plan_rejection_diagnostic(intent, exc)
            destination = self.state_root / "diagnostics" / ("plan-rejected-" + intent.operation_id + ".json")
            try:
                if destination.resolve().is_relative_to(Path(intent.root).resolve()):
                    raise ValueError("diagnostics must be outside library")
                atomic_json(destination, report)
                location = "diagnostic=diagnostics/" + destination.name
            except (OSError, ValueError):
                # Diagnostic failure must never turn a rejected plan into a run,
                # or bypass the retained-owner cleanup branch above.
                location = "diagnostic=unavailable"
            self._post(intent, "error", code + ":PlanRejected · " + summary + " · " + location)
        else:
            self._post(intent, "error", code + ":" + type(exc).__name__)

    def _drain_events(self):
        started, count = time.monotonic(), 0
        while count < 128 and time.monotonic() - started < .008:
            try:
                generation, operation_id, event = self._events.get_nowait()
            except queue.Empty:
                break
            count += 1
            if generation != self._generation or operation_id != self._op_id or self.state not in BUSY:
                continue
            if event["run_id"] != self.run_model.run_id:
                continue
            self.run_model.apply_event(event)
            if event["event"] in {"stage_changed", "track_finished", "track_failed", "track_cancelled", "track_not_started"}:
                self.log_message.emit(event["event"] + " · " + (event["stage"] or event["status"] or ""))

    def _receive(self, message):
        generation, operation_id, kind, value = message
        if generation != self._generation or operation_id != self._op_id:
            return
        if kind == "phase":
            if self.state != "STOPPING" or value == "RECONCILING":
                self._set_state(value)
        elif kind == "catalog":
            plan, directory, historical = value
            self.run_model.load_plan(plan)
            self.last_run_dir = directory
            if historical is not None:
                result, historical_directory = historical
                # Set historical run identity only for drawing, not for executable plan proofs.
                self.run_model.run_id = result["run_id"]
                display = dict(result, plan_hash=plan.plan_hash)
                self.run_model.apply_reconciliation(display)
                self.last_run_dir = historical_directory
                self._set_state("FINISHED")
                self.log_message.emit("目錄內容未變 · 已恢復上次完成結果（僅供顯示）")
            else:
                self._set_state("READY")
            self.catalog_changed.emit()
            self.log_message.emit("掃描完成 · " + str(len(plan.tracks)) + " Tracks")
        elif kind == "backend":
            self.last_backend = value
            self.backend_changed.emit(value)
            for backend in ("asr", "translation", "runtime", "conversion"):
                check = value.get(backend, {})
                self.log_message.emit("preflight · " + backend + " · " + str(check.get("error_code") or ("AVAILABLE" if check.get("available") else "UNAVAILABLE")))
        elif kind == "begin":
            plan, run_id, directory = value
            self.run_model.begin_run(plan, run_id)
            self.last_run_dir = directory
        elif kind == "delivery_gap":
            self.log_message.emit("事件傳遞不完整，將以權威收據與結果核對")
        elif kind == "snapshot":
            self.run_model.apply_snapshot(value)
        elif kind == "cleanup_failed":
            self._retrying = False
            self.last_cleanup = value
            self.last_error = "CLEANUP_FAILED"
            self._set_state("CLEANUP_FAILED")
            self.log_message.emit("清理未確認：保留所有權，請重試停止")
        elif kind == "finished":
            cleanup, result, directory = value
            # Drain queued notifications before applying the authoritative result.
            self._drain_events()
            self.run_model.apply_reconciliation(result)
            self.last_cleanup, self.last_run_dir = cleanup, directory
            self._save_historical_pointer(result, directory)
            self.log_message.emit("cleanup · verified=" + str(cleanup.get("cleanup_verified")) + " active=" + str(cleanup.get("active_count")))
            if cleanup.get("cleanup_verified") and result.get("run_id") and result.get("plan_hash"):
                record = (directory, result["run_id"], result["plan_hash"], self.root)
                if record not in self._trusted_runs:
                    self._trusted_runs.append(record)
            self._retrying = False
            self._set_state("FINISHED" if result.get("complete") else "ERROR")
            run_id = result.get("run_id")
            if run_id and run_id not in self._completed_ids:
                self._completed_ids.add(run_id)
                self.completed.emit(result)
            self.log_message.emit("批次結果 · " + str(result.get("outcome", "interrupted")))
        elif kind == "cancelled":
            self.last_cleanup = value
            self._set_state("READY" if self.run_model.plan else "IDLE")
        elif kind == "error":
            self.last_error = value
            self._set_state("ERROR")
            self.log_message.emit(value)
        if self._closing and not self.busy:
            self.close_ready.emit()
