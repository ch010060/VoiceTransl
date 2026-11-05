"""Small durable run store: authoritative receipts, snapshots, then JSONL notices."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import threading
import time
from uuid import uuid4

from .gui_cleanup import IntermediateCleanup, ROLES, WARNING as CLEANUP_WARNING
from .gui_plan import file_stamp, is_existing_subtitle_skip, make_provenance, path_key, validate_srt, Provenance
from .gui_protocol import (JSONLDecoder, ProtocolError, ReconciliationRequired,
                           TERMINAL, VERSION, check_version, uuid_text, validate_event)
from .gui_errors import ERROR_REASONS, curate_failure
from .gui_logging import DiagnosticLog, RunLogFormatter
from .srt_normalization import (normalize_srt_bytes, WARNING_CODE, MAX_BACKWARD_MS_PER_CUE,
                                MAX_REVERSED_CUES, MAX_REVERSED_PERCENT, FALLBACK_WARNING_CODE)


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def atomic_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as output:
            output.write(_json_bytes(payload) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError) as exc:
        raise ProtocolError("invalid stored JSON") from exc
    if not isinstance(data, dict):
        raise ProtocolError("stored JSON is not an object")
    return data


def safe_ref(run_dir, ref) -> Path:
    if not isinstance(ref, str) or not ref or "\\" in ref or ":" in ref or PureWindowsPath(ref).is_absolute():
        raise ProtocolError("invalid local reference")
    relative = Path(ref)
    if relative.is_absolute() or any(p in {"..", "."} for p in ref.split("/")):
        raise ProtocolError("reference escapes run directory")
    root = Path(run_dir).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ProtocolError("reference escapes run directory")
    return target


def _summary(tracks: dict) -> dict:
    counts = {status: 0 for status in sorted(TERMINAL)}
    for track in tracks.values():
        if track.get("status") in counts:
            counts[track["status"]] += 1
    return counts


def _run_outcome(counts, reasons, requested=False):
    if counts["failed"]:
        return "completed_with_errors", 1
    if counts["cancelled"] or counts["not_started"]:
        return ("cancelled" if requested or "run_cancelled" in reasons else "interrupted"), 2
    return "completed", 0


def _identity(track, mode, settings_hash) -> dict:
    role = "transcription" if mode == "transcribe_only" and track.task.action != "skip" else "final"
    output = track.task.transcribe_srt_path if role == "transcription" else track.task.final_srt_path
    if is_existing_subtitle_skip(track.task):
        role, output = "existing_subtitle", track.task.existing_subtitle_path
    return {"track_id": track.track_id, "work_id": track.work_id, "source_stamp": list(track.source_stamp),
            **({"shared_output_owner": track.shared_output_owner} if track.shared_output_owner else {}),
            "settings_hash": settings_hash, "output_type": role, "output_key": path_key(output),
            "verification": track.verification}


def _reason_code(reason):
    if reason is None:
        return None
    # Planner duplicate reasons include local relative filenames; keep these local.
    prefix = str(reason).split(":", 1)[0]
    return prefix if prefix in {"duplicate_audio", "existing_duplicate_subtitle", "existing_srt",
                                "existing_subtitle_skip", "dependency_failed", "run_cancelled", "run_aborted", "source_changed"} else "planned_action"


def _native_warnings(metadata, status):
    """Validate public native return metadata without retaining native output."""
    if metadata is None:
        return []
    if not isinstance(metadata, dict) or set(metadata) != {"backend", "native_exit_code", "returncode_known"}:
        raise ProtocolError("invalid native ASR metadata fields")
    code, known = metadata["native_exit_code"], metadata["returncode_known"]
    if metadata["backend"] != "xxl" or not isinstance(known, bool):
        raise ProtocolError("invalid native ASR metadata identity")
    if (known and (not isinstance(code, int) or isinstance(code, bool))) or (not known and code is not None):
        raise ProtocolError("invalid native ASR return code")
    warning = "ASR_NATIVE_EXIT_UNKNOWN" if not known else ("ASR_NATIVE_NONZERO_EXIT" if code != 0 else None)
    if warning and status in {"success", "transcribed", "copied", "skipped"}:
        raise ProtocolError("native ASR exit contradicts successful receipt")
    return [warning] if warning else []


def _cleanup_checkpoint(run_dir, manifest, state):
    path = safe_ref(run_dir, "intermediate_cleanup.json")
    if not path.exists():
        return {"schema_version": VERSION, "run_id": manifest["run_id"],
                "plan_hash": manifest["plan_hash"], "works": {}}
    record = read_json(path)
    _validate_identity(record, manifest["run_id"], manifest["plan_hash"])
    if not isinstance(record.get("works"), dict):
        raise ProtocolError("invalid work cleanup checkpoint")
    statuses = {"in_progress", "completed", "partial_failed", "deferred_dependencies", "deferred_interrupted"}
    files = {"pending_delete", "removed", "absent_after_intent", "failed_or_unsafe", "deferred_dependencies"}
    for work_id, work in record["works"].items():
        ids = {key for key, identity in manifest["tracks"].items() if identity["work_id"] == work_id}
        if (not ids or not isinstance(work, dict) or work.get("status") not in statuses
                or type(work.get("attempts")) is not int or not 0 <= work["attempts"] <= 3
                or not isinstance(work.get("tracks"), dict) or set(work["tracks"]) - ids
                or not isinstance(work.get("receipt_hashes"), dict)
                or not isinstance(work.get("waiting_for"), list)
                or any(key not in manifest["tracks"] for key in work["waiting_for"])
                or not ids <= work["receipt_hashes"].keys()
                or any(key not in state["tracks"] or state["tracks"][key].get("receipt_hash") != digest
                       for key, digest in work["receipt_hashes"].items())):
            raise ProtocolError("work cleanup receipt barrier mismatch")
        for entry in work["tracks"].values():
            if (not isinstance(entry, dict) or entry.get("status") not in
                    {"disabled", "completed", "partial_failed", "deferred_dependencies", "deferred_interrupted"}
                    or not isinstance(entry.get("files"), dict) or set(entry["files"]) - set(ROLES)
                    or any(value not in files for value in entry["files"].values())):
                raise ProtocolError("invalid intermediate cleanup state")
    return record


def cleanup_completed_work(run_dir, run_id, work_id, cleanup, *, retry=False):
    """Caller retains the original namespace lease; never acquire recovery authority."""
    plan = cleanup.plan
    run_dir = Path(run_dir).resolve()
    if run_dir.is_relative_to(plan.root):
        raise ProtocolError("cleanup records must be outside library")
    manifest, state = (read_json(safe_ref(run_dir, name)) for name in ("plan_identity.json", "state.json"))
    for record in (manifest, state):
        _validate_identity(record, run_id, plan.plan_hash)
    if manifest.get("settings_hash") != plan.settings_hash or manifest.get("mode") != plan.mode:
        raise ProtocolError("cleanup frozen settings mismatch")
    ids = [t.track_id for t in plan.planned_tracks if t.work_id == work_id]
    if not ids or any(state["tracks"][key].get("status") not in TERMINAL for key in ids):
        return None
    relevant = set(ids)
    paths = {p for key in ids for p in (plan.track_index[key].task.transcribe_srt_path,
                                       plan.track_index[key].task.translate_srt_path, plan.track_index[key].task.opencc_srt_path)}
    relevant.update(use.track_id for use in plan.output_ownership if use.path in paths and use.access == "read")
    relevant.update(t.track_id for t in plan.planned_tracks if set(t.dependencies) & set(ids))
    receipts = {}
    for key in relevant:
        stored = state["tracks"].get(key, {})
        if stored.get("status") in TERMINAL and stored.get("receipt_ref"):
            if manifest["tracks"].get(key) != _identity(plan.track_index[key], plan.mode, plan.settings_hash):
                raise ProtocolError("cleanup identity differs from frozen plan")
            receipts[key] = _verify_receipt(run_dir, manifest, key, stored)
    if not set(ids) <= receipts.keys():
        raise ProtocolError("cleanup lacks authoritative terminal receipts")
    try:
        checkpoint = _cleanup_checkpoint(run_dir, manifest, state)
    except OSError:
        return {"status": "persistence_failed", "tracks": {}}
    previous = checkpoint["works"].get(work_id)
    if previous and (previous["status"] == "completed" or previous["attempts"] >= 3
                     or not retry and previous["status"] != "deferred_dependencies"):
        return previous
    waiting = relevant - receipts.keys()
    entry = {"status": "in_progress", "attempts": (previous or {}).get("attempts", 0) + (not waiting),
             "receipt_hashes": {key: state["tracks"][key]["receipt_hash"] for key in receipts},
             "waiting_for": sorted(waiting), "tracks": (previous or {}).get("tracks", {})}
    checkpoint["works"][work_id] = entry
    def persist(records):
        entry["tracks"] = records
        atomic_json(safe_ref(run_dir, "intermediate_cleanup.json"), checkpoint)
    try:
        records = cleanup.run(ids, receipts, entry["tracks"], persist, interrupted=bool(
            state.get("cancel_requested") or any(
                receipts[key]["status"] == "cancelled" or
                (receipts[key]["status"] == "not_started" and receipts[key].get("reason") != "dependency_failed")
                for key in ids)))
        statuses = {record["status"] for record in records.values()}
        entry["status"] = ("completed" if statuses <= {"completed", "disabled"} else
                           "deferred_dependencies" if statuses == {"deferred_dependencies"} else
                           "deferred_interrupted" if statuses == {"deferred_interrupted"} else "partial_failed")
        persist(records)
    except OSError:
        entry["status"] = "persistence_failed"
    return entry


class RunStore:
    def __init__(self, run_dir, run_id, plan, stream, *, asr_metadata=None, ownership_check=None):
        self.run_dir = Path(run_dir).resolve()
        if self.run_dir.is_relative_to(plan.root.resolve()):
            raise ValueError("run records must be outside audio library")
        self.run_id = uuid_text(run_id)
        self.plan = plan
        self.stream = stream
        # The worker records native completion before issuing the terminal hook.
        # Keep the caller's mutable map: each Track acquires its result later.
        self.asr_metadata = asr_metadata if asr_metadata is not None else {}
        self.lock = threading.RLock()
        self.started_at = time.monotonic()
        self.started = False
        self.finished = False
        self.controls = {}
        self.reused_tracks = set()
        self.normalizations = {}
        self.intermediate_cleanup = IntermediateCleanup(plan, ownership_check)
        self._work_cleanup, self._pending_cleanup = {}, {}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if any(p.name not in {"plan.json"} for p in self.run_dir.iterdir()):
            raise FileExistsError("run store is not new")
        with (self.run_dir / ".store").open("x", encoding="ascii") as marker:
            marker.write(self.run_id)
        (self.run_dir / "receipts").mkdir()
        (self.run_dir / "logs").mkdir()
        self.identities = {t.track_id: _identity(t, plan.mode, plan.settings_hash) for t in plan.planned_tracks}
        self.manifest = {"schema_version": VERSION, "run_id": self.run_id, "plan_hash": plan.plan_hash,
                         "settings_hash": plan.settings_hash, "mode": plan.mode, "tracks": self.identities}
        atomic_json(self.run_dir / "plan_identity.json", self.manifest)
        self.state = {"schema_version": VERSION, "run_id": self.run_id, "plan_hash": plan.plan_hash,
                      "last_seq": 0, "status": "ready", "tracks": {
                          t.track_id: {"work_id": t.work_id, "revision": 0, "status": None}
                          for t in plan.planned_tracks}, "controls": {}}
        self._save_state()
        self._readable = DiagnosticLog(self.run_dir / "logs" / "execution.log")
        self._log_formatter = RunLogFormatter(plan, self.run_id, self.run_dir)

    def _save_state(self):
        self.state["counts"] = _summary(self.state["tracks"])
        atomic_json(self.run_dir / "state.json", self.state)

    def record_normalization(self, track_id, raw, normalized, changes):
        """Private, run-owned evidence prepared before replacing a fresh ASR file.

        Retained with the run (including failed/cancelled attempts); never copied
        to the audio library or public event stream. No independent cleanup.
        """
        with self.lock:
            track = self.plan.track_index[track_id]
            if (track_id not in self.identities or track_id in self.normalizations
                    or track.task.action != "process" or track.shared_output_owner
                    or self.state["tracks"][track_id].get("status") != "running"
                    or self.run_dir.is_relative_to(self.plan.root.resolve())
                    or dict(self.plan.inventory)[track.task.transcribe_srt_path][-1] is not None
                    or file_stamp(track.task.audio_path) != track.source_stamp):
                raise ProtocolError("normalization is not a fresh owned ASR output")
            expected, expected_changes = normalize_srt_bytes(raw)
            if not changes or expected != normalized or expected_changes != changes:
                raise ProtocolError("invalid normalization evidence")
            prefix = "logs/srt-normalization/" + track_id
            audit = {"code": WARNING_CODE, "track_id": track_id, "plan_hash": self.plan.plan_hash,
                     "algorithm": "start-order-v1",
                     "policy": {"max_backward_ms_per_cue": MAX_BACKWARD_MS_PER_CUE,
                                "max_reversed_cues": MAX_REVERSED_CUES,
                                "max_reversed_percent": MAX_REVERSED_PERCENT},
                     "source_stamp": list(track.source_stamp), "settings_hash": self.plan.settings_hash,
                     "transcription_key": path_key(track.task.transcribe_srt_path),
                     "raw_ref": prefix + ".raw.srt", "normalized_ref": prefix + ".normalized.srt",
                     "raw_hash": hashlib.sha256(raw).hexdigest(),
                     "normalized_hash": hashlib.sha256(normalized).hexdigest(),
                     "repaired_cues": len(changes),
                     "max_displacement_ms": max(c["displacement_ms"] for c in changes),
                     "total_displacement_ms": sum(c["displacement_ms"] for c in changes),
                     "changes": changes}
            if changes[0].get("fallback"):
                audit["fallback"] = changes[0]["fallback"]
            for ref, data in ((audit["raw_ref"], raw), (audit["normalized_ref"], normalized)):
                path = safe_ref(self.run_dir, ref)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
            ref = prefix + ".json"
            path = safe_ref(self.run_dir, ref)
            if path.exists():
                raise ProtocolError("normalization audit already exists")
            atomic_json(path, audit)
            self.normalizations[track_id] = {"ref": ref, "hash": _digest(audit)}

    def emit(self, event, *, work_id=None, track_id=None, stage=None, status=None, payload=None) -> dict:
        with self.lock:
            if self.finished:
                raise ProtocolError("run already finished")
            track = self.state["tracks"].get(track_id) if track_id else None
            if track_id and (track is None or track["status"] in TERMINAL):
                raise ProtocolError("unknown or terminal Track")
            if track:
                if work_id is not None and work_id != track["work_id"]:
                    raise ProtocolError("Track Work mismatch")
                work_id = track["work_id"]
            record = {"schema_version": VERSION, "event": event, "run_id": self.run_id,
                      "seq": self.state["last_seq"] + 1,
                      "emitted_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                      "elapsed_ms": int((time.monotonic() - self.started_at) * 1000),
                      "work_id": work_id, "track_id": track_id,
                      "track_revision": track["revision"] + 1 if track else None,
                      "stage": stage, "status": status, "payload": payload or {}}
            validate_event(record, self.run_id)
            encoded = _json_bytes(record) + b"\n"
            if len(encoded) > 64 * 1024:
                raise ProtocolError("event exceeds frame bound")
            if track:
                track.update(revision=record["track_revision"], status=status, stage=stage)
            self.state["last_seq"] = record["seq"]
            self._save_state()
            with (self.run_dir / "events.jsonl").open("ab") as journal:
                journal.write(encoded)
                journal.flush()
                os.fsync(journal.fileno())
            self.stream.write(encoded.decode("utf-8"))
            self.stream.flush()
            # Human logs are optional presentation; protocol/receipt I/O above
            # remains authoritative and must still fail closed.
            try:
                for message, context in self._log_formatter.event(record):
                    self._readable.append(self.plan.root, message, context)
            except Exception:
                pass
            return record

    def start(self) -> None:
        with self.lock:
            if self.started:
                raise ProtocolError("run already started")
            self.started = True
            self.state["status"] = "running"
            self.emit("run_started", status="running", payload={"plan_hash": self.plan.plan_hash,
                      "config_hash": self.plan.settings_hash, "mode": self.plan.mode,
                      "planned_tracks": len(self.plan.planned_tracks)})
            planned = set(self.identities)
            for work in self.plan.works:
                self.emit("work_discovered", work_id=work.work_id, payload={"kind": work.kind,
                          "track_count": len(work.track_ids), "actionable_count": sum(t in planned for t in work.track_ids),
                          "catalog_ref": "plan:work:" + work.work_id})

    def _terminal_receipt(self, event, track):
        if track.shared_output_owner and event.status in {"success", "transcribed", "skipped"}:
            raise ProtocolError("shared output reader must report copied")
        identity = self.identities[track.track_id]
        provenance = None
        output_hash = None
        warnings = []
        raw_metadata = self.asr_metadata.get(track.track_id)
        native_metadata = None
        if raw_metadata is not None:
            if not isinstance(raw_metadata, dict):
                raise ProtocolError("invalid native ASR metadata")
            native_metadata = {key: raw_metadata.get(key) for key in ("backend", "native_exit_code", "returncode_known")}
        native_warnings = _native_warnings(native_metadata, event.status)
        if event.status in {"success", "transcribed", "copied"}:
            if identity["output_type"] == "existing_subtitle":
                raise ProtocolError("existing subtitle cannot certify a completed stage")
            if (event.status == "success" and self.plan.mode != "full" or
                    event.status == "transcribed" and self.plan.mode != "transcribe_only"):
                raise ProtocolError("terminal outcome contradicts planned mode")
            output = Path(identity["output_key"])
            provenance = asdict(make_provenance(self.plan, track.track_id, output, identity["output_type"]))
            output_hash = provenance["output_hash"]
        elif event.status == "skipped":
            output = Path(identity["output_key"])
            if identity["output_type"] == "existing_subtitle":
                source = track.task.existing_subtitle_path
                stamp = dict(self.plan.inventory).get(source)
                if not output.is_file() or stamp is None or file_stamp(source) != stamp:
                    raise ProtocolError("existing subtitle changed after planning")
            else:
                validate_srt(output)
            output_hash = hashlib.sha256(output.read_bytes()).hexdigest()
            if identity["output_type"] == "existing_subtitle" and output_hash != stamp[-1]:
                raise ProtocolError("existing subtitle changed while recording skip")
            # A skip records an existing file, never certifies a completed stage.
            warnings = ["unverified_legacy"] if track.verification == "unverified_legacy" else []
        warnings.extend(native_warnings)
        normalization = self.normalizations.get(track.track_id)
        for warning in (WARNING_CODE, FALLBACK_WARNING_CODE):
            if warning in event.warnings:
                if normalization is None:
                    raise ProtocolError("normalization warning lacks controlled evidence")
                warnings.append(warning)
        receipt = {"schema_version": VERSION, "run_id": self.run_id, "plan_hash": self.plan.plan_hash,
                   **identity, "status": event.status, "output_hash": output_hash,
                   "provenance": provenance, "warnings": warnings,
                   "native_metadata": native_metadata,
                   "reason": _reason_code(event.reason),
                   "error_code": "TRACK_EXECUTION_FAILED" if event.status == "failed" else None,
                   "dependencies": list(track.dependencies),
                   "revision": self.state["tracks"][track.track_id]["revision"] + 1}
        if event.failure is not None or event.status == "failed":
            error = {**curate_failure(event.failure, event.stage), "work_id": track.work_id,
                     "track_id": track.track_id, "outcome": event.status,
                     "audio_path": track.task.audio_path.relative_to(track.task.work_dir).as_posix()}
            ref = "logs/" + track.track_id + ".error.json"
            error["diagnostic_ref"] = ref
            try:
                atomic_json(safe_ref(self.run_dir, ref), error)
            except OSError:
                error["diagnostic_ref"] = "receipts/" + track.track_id + ".json"
            receipt.update(error=error, error_code=error["code"])
        if normalization is not None:
            receipt["timestamp_normalization"] = normalization
            _verify_normalization(self.run_dir, receipt)
        if event.status == "success" and track.task.action == "process" and not track.shared_output_owner:
            receipt["intermediate_artifacts"] = self.intermediate_cleanup.evidence(track.track_id)
        if event.status == "failed" and str(event.reason).startswith("SRT_TIMESTAMP_ORDER_INVALID:"):
            receipt["error_code"] = "SRT_TIMESTAMP_ORDER_INVALID"
            receipt["timestamp_error"] = event.reason.split(":", 1)[1]
        if track.shared_output_owner and event.status == "copied":
            owner_state = self.state["tracks"][track.shared_output_owner]
            if not owner_state.get("receipt_ref"):
                raise ProtocolError("shared output owner lacks a terminal receipt")
            owner_receipt = _verify_receipt(self.run_dir, self.manifest, track.shared_output_owner, owner_state)
            if (owner_receipt["status"] not in {"success", "transcribed", "copied", "skipped"}
                    or owner_receipt["output_key"] != identity["output_key"]
                    or owner_receipt["output_type"] != identity["output_type"]
                    or owner_receipt["output_hash"] != output_hash):
                raise ProtocolError("shared output differs from owner receipt")
            receipt["shared_output_receipt"] = {key: owner_state[key] for key in
                                                ("receipt_ref", "receipt_hash", "status", "revision")}
        ref = "receipts/" + track.track_id + ".json"
        atomic_json(safe_ref(self.run_dir, ref), receipt)
        self.state["tracks"][track.track_id].update(receipt_ref=ref, receipt_hash=_digest(receipt))
        return ref, receipt

    def observer(self, event) -> None:
        with self.lock:
            track = self.plan.track_index.get(event.track_id)
            if track is None or event.track_id not in self.identities:
                raise ProtocolError("event outside frozen plan")
            name, stage, status = event.event, event.stage, event.status
            if name == "asr_reused":
                self.reused_tracks.add(track.track_id)
                return
            if name == "asr_completed" or (name == "stage_changed" and stage == "skip"):
                return
            payload = {}
            if name == "track_queued":
                payload = {"action": track.task.action, "reason": _reason_code(track.task.reason),
                           "ordinal": track.ordinal, "dependencies": list(track.dependencies)}
            elif name == "track_started":
                payload = {"action": track.task.action}
            elif name == "stage_changed":
                payload = {"progress_mode": "indeterminate"}
                if track.track_id in self.reused_tracks:
                    payload["reused"] = True
            elif name == "asr_progress":
                ratio, current, total = event.progress or (None, None, None)
                if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in (ratio, current, total)) or total <= 0:
                    raise ProtocolError("invalid native ASR progress")
                payload = {"reported_ratio": max(0, min(1, ratio)), "current_audio_s": current,
                           "total_audio_s": total, "source": "xxl_segment_start"}
            elif name in {"track_finished", "track_failed", "track_cancelled", "track_not_started"}:
                if status not in TERMINAL:
                    raise ProtocolError("invalid terminal outcome")
                ref, receipt = self._terminal_receipt(event, track)
                payload = {"receipt_id": ref}
                stage = None
                if name == "track_finished":
                    payload.update(outcome=status, output_ids=[receipt["output_type"]], warnings=receipt["warnings"])
                elif name == "track_failed":
                    payload["error"] = {"code": receipt["error_code"], "retryable": True,
                                        "detail_ref": receipt.get("error", {}).get("diagnostic_ref", ref)}
                else:
                    payload["reason"] = event.reason if event.reason in {"dependency_failed", "run_cancelled", "run_aborted"} else "run_aborted"
                    if name == "track_cancelled":
                        payload["checkpoint_refs"] = []
            self.emit(name, track_id=track.track_id, stage=stage, status=status, payload=payload)
            if status in TERMINAL:
                ready = {track.work_id} | {key for key, waiting in self._pending_cleanup.items() if track.track_id in waiting}
                for work_id in ready:
                    self.cleanup_work(work_id)

    def cleanup_work(self, work_id):
        with self.lock:
            ids = [t.track_id for t in self.plan.planned_tracks if t.work_id == work_id]
            if any(self.state["tracks"][key]["status"] not in TERMINAL for key in ids):
                return
            entry = cleanup_completed_work(self.run_dir, self.run_id, work_id, self.intermediate_cleanup)
            if entry is not None:
                self._work_cleanup[work_id] = entry
                if entry.get("waiting_for"):
                    self._pending_cleanup[work_id] = set(entry["waiting_for"])
                else:
                    self._pending_cleanup.pop(work_id, None)
            return entry

    def snapshot(self) -> dict:
        with self.lock:
            self._save_state()
            return json.loads(_json_bytes(self.state))

    def ack_control(self, message, token) -> None:
        with self.lock:
            if not isinstance(message, dict) or not {"schema_version", "run_id", "request_id", "command", "payload"} <= message.keys():
                raise ProtocolError("invalid control envelope")
            check_version(message["schema_version"])
            uuid_text(message["request_id"])
            if message["run_id"] != self.run_id or message["command"] not in {"cancel", "request_snapshot"} or not isinstance(message["payload"], dict):
                raise ProtocolError("invalid control identity or command")
            request = message["request_id"]
            if request in self.controls:
                if self.controls[request] != message:
                    raise ProtocolError("conflicting control request")
                return
            command = message["command"]
            if command == "cancel":
                if not self.state.get("cancel_requested"):
                    token.cancel()
                self.state.update(cancel_requested=True, status="cancelling")
            self.controls[request] = dict(message)
            self.state["controls"][request] = command
            self._save_state()
            payload = {"request_id": request, "command": command, "accepted": True}
            if command == "request_snapshot":
                payload["snapshot_ref"] = "state.json"
            self.emit("control_ack", payload=payload)

    def finish(self, results) -> dict:
        with self.lock:
            if self.finished:
                return read_json(self.run_dir / "result.json")
            by_id = {r.track_id: r for r in results}
            if len(by_id) != len(results) or set(by_id) != set(self.identities):
                raise ProtocolError("results do not cover frozen plan")
            for track_id, result in by_id.items():
                stored = self.state["tracks"][track_id]
                if result.status not in TERMINAL or stored["status"] != result.status or not stored.get("receipt_ref"):
                    raise ProtocolError("result differs from authoritative terminal receipt")
                receipt = _verify_receipt(self.run_dir, self.manifest, track_id, stored)
                if result.status in {"success", "transcribed", "copied", "skipped"}:
                    if result.output_path is None or path_key(result.output_path) != self.identities[track_id]["output_key"]:
                        raise ProtocolError("result output differs from planned receipt")
                if result.status in {"success", "transcribed", "copied"}:
                    # Shared representative copies intentionally have no 1A
                    # Provenance entry; the trusted terminal callback certifies
                    # that mode-specific read, verified above against its owner.
                    proofs = getattr(result, "provenance", ())
                    terminal_proofs = [json.loads(_json_bytes(asdict(proof))) for proof in proofs
                                       if proof.output_type == receipt["output_type"]]
                    if terminal_proofs and receipt["provenance"] not in terminal_proofs:
                        raise ProtocolError("result provenance differs from terminal receipt")
            counts = _summary(self.state["tracks"])
            reasons = [read_json(safe_ref(self.run_dir, t["receipt_ref"])).get("reason") for t in self.state["tracks"].values()]
            outcome, code = _run_outcome(counts, reasons, self.state.get("cancel_requested", False))
            result = {"schema_version": VERSION, "run_id": self.run_id, "plan_hash": self.plan.plan_hash,
                      "outcome": outcome, "counts": counts, "exit_code": code, "complete": True,
                      "tracks": {k: {key: value[key] for key in ("status", "receipt_ref", "receipt_hash", "revision")}
                                 for k, value in self.state["tracks"].items()}}
            for work_id in {t.work_id for t in self.plan.planned_tracks} - self._work_cleanup.keys():
                self.cleanup_work(work_id)
            cleanup = {key: record for work in self._work_cleanup.values() for key, record in work["tracks"].items()}
            result["intermediate_cleanup"] = cleanup
            result["warnings"] = [CLEANUP_WARNING] if any(
                work["status"] != "completed" for work in self._work_cleanup.values()) else []
            atomic_json(self.run_dir / "result.json", result)
            self.state["status"] = outcome
            self.emit("run_finished", status=outcome, payload={"outcome": outcome, "counts": counts,
                      "result_ref": "result.json", "result_hash": _digest(result)})
            self.finished = True
            try:
                for message, context in self._log_formatter.reconcile(result, {}):
                    self._readable.append(self.plan.root, message, context)
            except Exception:
                pass
            self._readable.close(wait=True)
            return result


def _validate_identity(record, run_id, plan_hash):
    check_version(record.get("schema_version"))
    if record.get("run_id") != run_id or record.get("plan_hash") != plan_hash:
        raise ProtocolError("stored run or plan identity mismatch")


def _verify_normalization(run_dir, receipt):
    link = receipt.get("timestamp_normalization")
    if link is None:
        if any(w in receipt["warnings"] for w in (WARNING_CODE, FALLBACK_WARNING_CODE)):
            raise ProtocolError("normalization warning lacks evidence")
        return
    if not isinstance(link, dict) or set(link) != {"ref", "hash"}:
        raise ProtocolError("invalid normalization reference")
    if receipt["status"] in {"success", "transcribed"} and WARNING_CODE not in receipt["warnings"]:
        raise ProtocolError("normalization receipt warning missing")
    try:
        audit = read_json(safe_ref(run_dir, link["ref"]))
    except OSError as exc:
        raise ProtocolError("normalization audit missing") from exc
    if (_digest(audit) != link["hash"] or audit.get("code") != WARNING_CODE
            or audit.get("algorithm") != "start-order-v1"
            or audit.get("policy") != {"max_backward_ms_per_cue": MAX_BACKWARD_MS_PER_CUE,
                                     "max_reversed_cues": MAX_REVERSED_CUES,
                                     "max_reversed_percent": MAX_REVERSED_PERCENT}
            or any(audit.get(k) != receipt[k] for k in ("track_id", "plan_hash", "source_stamp", "settings_hash"))):
        raise ProtocolError("normalization audit identity or hash mismatch")
    try:
        raw = safe_ref(run_dir, audit["raw_ref"]).read_bytes()
        normalized = safe_ref(run_dir, audit["normalized_ref"]).read_bytes()
        expected, changes = normalize_srt_bytes(raw)
        if (not changes or expected != normalized or changes != audit["changes"]
                or hashlib.sha256(raw).hexdigest() != audit["raw_hash"]
                or hashlib.sha256(normalized).hexdigest() != audit["normalized_hash"]
                or len(changes) != audit["repaired_cues"]
                or max(c["displacement_ms"] for c in changes) != audit["max_displacement_ms"]
                or sum(c["displacement_ms"] for c in changes) != audit["total_displacement_ms"]):
            raise ValueError("normalization bytes or summary differ")
        fallback = changes[0].get("fallback")
        if (_digest(audit.get("fallback")) != _digest(fallback)
                or (FALLBACK_WARNING_CODE in receipt["warnings"] and (not fallback or WARNING_CODE not in receipt["warnings"]))
                or (fallback and WARNING_CODE in receipt["warnings"] and FALLBACK_WARNING_CODE not in receipt["warnings"])):
            raise ValueError("normalization fallback evidence or warning differs")
        validate_srt(safe_ref(run_dir, audit["normalized_ref"]))
    except (KeyError, OSError, ValueError) as exc:
        raise ProtocolError("normalization evidence missing or invalid") from exc


def _verify_receipt(run_dir, manifest, track_id, stored):
    receipt = read_json(safe_ref(run_dir, stored["receipt_ref"]))
    _validate_identity(receipt, manifest["run_id"], manifest["plan_hash"])
    expected = manifest["tracks"][track_id]
    if _digest(receipt) != stored.get("receipt_hash") or any(receipt.get(k) != v for k, v in expected.items()):
        raise ProtocolError("receipt identity or hash mismatch")
    if receipt.get("status") != stored.get("status") or receipt.get("revision") != stored.get("revision") or receipt.get("status") not in TERMINAL:
        raise ProtocolError("receipt revision or outcome mismatch")
    status = receipt["status"]
    artifacts = receipt.get("intermediate_artifacts")
    if artifacts is not None:
        if (status != "success" or not isinstance(artifacts, dict) or set(artifacts) - set(ROLES)
                or any(not isinstance(value, dict) or set(value) != {"stamp", "directories"}
                       or not isinstance(value["stamp"], list) or len(value["stamp"]) != 6
                       or not isinstance(value["directories"], list) or not value["directories"]
                       or any(not isinstance(d, list) or len(d) != 3 or not isinstance(d[0], str)
                              or any(type(n) is not int for n in d[1:]) for d in value["directories"])
                       for value in artifacts.values())):
            raise ProtocolError("invalid intermediate artifact evidence")
    error = receipt.get("error")
    if error is not None:
        if (not isinstance(error, dict) or error.get("work_id") != expected["work_id"]
                or error.get("track_id") != track_id or error.get("outcome") != status
                or error.get("code") not in ERROR_REASONS or receipt.get("error_code") != error["code"]
                or error.get("reason") != ERROR_REASONS[error["code"]]):
            raise ProtocolError("receipt failure identity mismatch")
        safe_ref(run_dir, error.get("diagnostic_ref"))
    existing_subtitle = expected.get("output_type") == "existing_subtitle"
    if existing_subtitle and status in {"success", "copied", "transcribed"}:
        raise ProtocolError("existing subtitle cannot certify a completed stage")
    native_warnings = _native_warnings(receipt.get("native_metadata"), status)
    if not isinstance(receipt.get("warnings"), list) or any(warning not in receipt["warnings"] for warning in native_warnings):
        raise ProtocolError("native ASR receipt warning missing")
    _verify_normalization(run_dir, receipt)
    if existing_subtitle and (receipt.get("provenance") is not None or
                              (status == "skipped" and "unverified_legacy" not in receipt["warnings"])):
        raise ProtocolError("existing subtitle receipt claims verification")
    if status in {"success", "copied", "transcribed", "skipped"}:
        if list(file_stamp(Path(expected["source_stamp"][0]))) != expected["source_stamp"]:
            raise ProtocolError("receipt source inventory changed")
        output = Path(expected["output_key"])
        try:
            if existing_subtitle:
                if not output.is_file():
                    raise ValueError("existing subtitle is not a file")
            else:
                validate_srt(output)
            actual_hash = hashlib.sha256(output.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            raise ProtocolError("receipt output missing or invalid") from exc
        if actual_hash != receipt.get("output_hash"):
            raise ProtocolError("receipt output hash mismatch")
        if status != "skipped":
            proof = receipt.get("provenance")
            expected_proof = {k: expected[k] for k in ("track_id", "source_stamp", "settings_hash", "output_key", "output_type")}
            expected_proof["output_hash"] = actual_hash
            if proof != expected_proof:
                raise ProtocolError("receipt provenance mismatch")
        owner_id = expected.get("shared_output_owner")
        if owner_id:
            owner = manifest["tracks"].get(owner_id)
            link = receipt.get("shared_output_receipt")
            if (status != "copied" or not owner or owner.get("shared_output_owner")
                    or owner["work_id"] != expected["work_id"]
                    or owner["output_key"] != expected["output_key"]
                    or owner["output_type"] != expected["output_type"]
                    or receipt.get("dependencies") != [owner_id] or not isinstance(link, dict)
                    or set(link) != {"receipt_ref", "receipt_hash", "status", "revision"}
                    or link["receipt_ref"] != "receipts/" + owner_id + ".json"):
                raise ProtocolError("invalid shared output receipt dependency")
            owner_receipt = _verify_receipt(run_dir, manifest, owner_id, link)
            if (owner_receipt["status"] not in {"success", "transcribed", "copied", "skipped"}
                    or owner_receipt["output_hash"] != actual_hash):
                raise ProtocolError("shared output owner did not complete")
    return receipt


def reconcile_run(run_dir, run_id, plan_hash, exit_code=None, *, verify_journal=True) -> dict:
    """Read-only reconciliation. Controller records recovery only after cleanup."""
    run_dir = Path(run_dir).resolve()
    uuid_text(run_id)
    manifest = read_json(safe_ref(run_dir, "plan_identity.json"))
    state = read_json(safe_ref(run_dir, "state.json"))
    _validate_identity(manifest, run_id, plan_hash)
    _validate_identity(state, run_id, plan_hash)
    if set(state.get("tracks", {})) != set(manifest.get("tracks", {})):
        raise ProtocolError("snapshot Track set differs from frozen plan")
    recovered = {}
    for track_id, stored in state["tracks"].items():
        # Receipt publication can survive a crash before the following snapshot.
        orphan_ref = "receipts/" + track_id + ".json"
        orphan_path = safe_ref(run_dir, orphan_ref)
        if not stored.get("receipt_ref") and orphan_path.exists():
            orphan = read_json(orphan_path)
            if orphan.get("revision") != stored.get("revision", 0) + 1:
                raise ProtocolError("orphan receipt revision mismatch")
            stored = {**stored, "receipt_ref": orphan_ref, "receipt_hash": _digest(orphan),
                      "status": orphan.get("status"), "revision": orphan.get("revision")}
        if stored.get("receipt_ref"):
            _verify_receipt(run_dir, manifest, track_id, stored)
            recovered[track_id] = dict(stored)
        elif stored.get("status") in TERMINAL:
            raise ProtocolError("terminal snapshot lacks authoritative receipt")
    counts = _summary(recovered)
    warnings = []
    intermediate_cleanup = {}
    try:
        checkpoint = _cleanup_checkpoint(run_dir, manifest, {**state, "tracks": {**state["tracks"], **recovered}})
        intermediate_cleanup = {key: value for work in checkpoint["works"].values() for key, value in work["tracks"].items()}
        if any(work["status"] != "completed" for work in checkpoint["works"].values()):
            warnings.append(CLEANUP_WARNING)
    except OSError:
        warnings.append(CLEANUP_WARNING)
    decoder = JSONLDecoder(run_id)
    terminal_notice = None
    journal_events = []
    try:
        if verify_journal:
            with safe_ref(run_dir, "events.jsonl").open("rb") as journal:
                while chunk := journal.read(64 * 1024):
                    for event in decoder.feed(chunk):
                        journal_events.append(event)
                        if event["event"] == "run_finished":
                            terminal_notice = event
            decoder.finish()
    except (ReconciliationRequired, OSError):
        warnings.append("JOURNAL_RECONCILIATION_REQUIRED")
    except ProtocolError as exc:
        if str(exc) not in {"truncated JSONL frame", "malformed JSONL frame"}:
            raise
        warnings.append("JOURNAL_RECONCILIATION_REQUIRED")
    for event in journal_events:
        track_id = event["track_id"]
        if track_id and (track_id not in manifest["tracks"] or event["work_id"] != manifest["tracks"][track_id]["work_id"]):
            raise ProtocolError("journal identity outside frozen plan")
        payload = event["payload"]
        refs = [payload[k] for k in ("receipt_id", "result_ref", "snapshot_ref") if k in payload]
        if isinstance(payload.get("error"), dict):
            refs.append(payload["error"]["detail_ref"])
        for ref in refs:
            safe_ref(run_dir, ref)
    if verify_journal and decoder.last_seq != state.get("last_seq"):
        warnings.append("JOURNAL_SNAPSHOT_DIFFER")
    result_path = safe_ref(run_dir, "result.json")
    result = None
    failure_result = False
    if result_path.exists():
        result = read_json(result_path)
        _validate_identity(result, run_id, plan_hash)
        receipt_tracks = {
                k: {key: value[key] for key in ("status", "receipt_ref", "receipt_hash", "revision")}
                for k, value in recovered.items()}
        failure_codes = {"preflight_failed": 3, "plan_rejected": 3, "protocol_failed": 4, "internal_error": 4}
        if result.get("outcome") in failure_codes:
            failure_result = True
            error = result.get("error")
            if result.get("complete") is not False or result.get("exit_code") != failure_codes[result["outcome"]] or not isinstance(error, dict) or not isinstance(error.get("code"), str):
                raise ProtocolError("invalid startup or fatal failure result")
            if result["outcome"] in {"preflight_failed", "plan_rejected"} and recovered:
                raise ProtocolError("startup failure contains terminal work")
        # A fatal result may have captured the snapshot immediately before a
        # receipt publication; reconstructed receipt counts are authoritative.
        allowed_counts = [counts, state.get("counts")] if failure_result else [counts]
        if result.get("counts") not in allowed_counts or ("tracks" in result and result["tracks"] != receipt_tracks):
            raise ProtocolError("result differs from authoritative receipts")
        if not failure_result and ("tracks" not in result or len(recovered) != len(manifest["tracks"])):
            raise ProtocolError("result lacks planned outcomes")
        reasons = [read_json(safe_ref(run_dir, t["receipt_ref"])).get("reason") for t in recovered.values()]
        expected = _run_outcome(counts, reasons, state.get("cancel_requested", False))
        if not failure_result and (result.get("outcome"), result.get("exit_code")) != expected:
            raise ProtocolError("result success or exit code contradicts outcomes")
        if terminal_notice and (terminal_notice["payload"].get("result_ref") != "result.json" or
                                terminal_notice["payload"].get("result_hash") != _digest(result)):
            raise ProtocolError("published result hash mismatch")
    if result is not None and "intermediate_cleanup" in result:
        cleanup = result["intermediate_cleanup"]
        if (not isinstance(cleanup, dict) or any(k not in recovered for k in cleanup)
                or result.get("warnings") not in ([], [CLEANUP_WARNING])):
            raise ProtocolError("invalid intermediate cleanup result")
        warnings.extend(result["warnings"])
        if not intermediate_cleanup:
            intermediate_cleanup = result["intermediate_cleanup"]
    complete = result is not None and not failure_result
    outcome = result["outcome"] if result is not None else "interrupted"
    if exit_code is not None and (exit_code != 0 and outcome == "completed" or exit_code == 0 and result is None):
        outcome, complete = "interrupted", False
        warnings.append("WORKER_EXIT_RESULT_MISMATCH")
    return {"schema_version": VERSION, "run_id": run_id, "plan_hash": plan_hash,
            "outcome": outcome, "complete": complete, "counts": counts,
            "tracks": recovered, "missing_track_ids": sorted(set(manifest["tracks"]) - set(recovered)),
            "warnings": list(dict.fromkeys(warnings)), "intermediate_cleanup": intermediate_cleanup,
            "exit_code": exit_code if exit_code is not None else (result or {}).get("exit_code")}


def resume_provenance(run_dir, run_id, plan_hash) -> tuple[Provenance, ...]:
    reconciled = reconcile_run(run_dir, run_id, plan_hash)
    proofs = []
    for stored in reconciled["tracks"].values():
        receipt = read_json(safe_ref(run_dir, stored["receipt_ref"]))
        proof = receipt.get("provenance")
        if proof:
            proof["source_stamp"] = tuple(proof["source_stamp"])
            proofs.append(Provenance(**proof))
    return tuple(proofs)
