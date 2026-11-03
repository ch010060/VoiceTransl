"""Bounded v1 JSONL framing and ordering for the isolated ASMR worker."""
from __future__ import annotations

import json
import math
from datetime import datetime
from uuid import UUID

VERSION = "1.0"
MAX_LINE = 64 * 1024
EVENTS = {
    "worker_ready", "run_started", "work_discovered", "track_queued",
    "track_started", "stage_changed", "asr_progress", "track_finished",
    "track_failed", "track_cancelled", "track_not_started", "heartbeat",
    "control_ack", "run_finished",
}
STAGES = {
    "planning", "preflight", "waiting_dependency", "waiting_asr", "asr",
    "transcript_correction", "translation", "opencc", "traditional_correction",
    "finalize", "copy",
}
TERMINAL = {"success", "skipped", "copied", "transcribed", "failed", "cancelled", "not_started"}
FIELDS = {"schema_version", "event", "run_id", "seq", "emitted_at", "elapsed_ms",
          "work_id", "track_id", "track_revision", "stage", "status", "payload"}
PAYLOAD_FIELDS = {
    "worker_ready": {"protocol", "containment_ready"},
    "run_started": {"plan_hash", "config_hash", "mode", "planned_tracks"},
    "work_discovered": {"kind", "track_count", "actionable_count", "catalog_ref"},
    "track_queued": {"action", "reason", "ordinal", "dependencies"},
    "track_started": {"action"},
    "stage_changed": {"reused", "progress_mode"},
    "asr_progress": {"reported_ratio", "current_audio_s", "total_audio_s", "source", "warnings"},
    "track_finished": {"outcome", "output_ids", "receipt_id", "warnings"},
    "track_failed": {"error", "receipt_id"},
    "track_cancelled": {"reason", "checkpoint_refs", "receipt_id"},
    "track_not_started": {"reason", "receipt_id"},
    "heartbeat": {"last_seq", "active_count"},
    "control_ack": {"request_id", "command", "accepted", "snapshot_ref"},
    "run_finished": {"outcome", "counts", "result_ref", "result_hash"},
}
REQUIRED_PAYLOAD = {
    **PAYLOAD_FIELDS,
    "stage_changed": {"progress_mode"},
    "asr_progress": {"reported_ratio", "current_audio_s", "total_audio_s", "source"},
    "track_failed": {"error"},
    "track_cancelled": {"reason", "checkpoint_refs"},
    "track_not_started": {"reason"},
    "control_ack": {"request_id", "command", "accepted"},
}


class ProtocolError(ValueError):
    """Malformed, incompatible or conflicting protocol data."""


class ReconciliationRequired(ProtocolError):
    """A valid frame cannot be applied until durable state is reconciled."""


def uuid_text(value: object) -> str:
    if not isinstance(value, str):
        raise ProtocolError("invalid UUID")
    try:
        UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError("invalid UUID") from exc
    return value


def check_version(value: object) -> None:
    if not isinstance(value, str) or len(value.split(".")) != 2 or not all(p.isdigit() for p in value.split(".")) or value.split(".")[0] != "1":
        raise ProtocolError("unsupported schema version")


def _integer(value: object, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def validate_event(event: object, run_id: str) -> dict:
    if not isinstance(event, dict) or not FIELDS <= event.keys():
        raise ProtocolError("missing event envelope fields")
    check_version(event["schema_version"])
    uuid_text(event["run_id"])
    if event["run_id"] != run_id:
        raise ProtocolError("stale run identity")
    name = event["event"]
    if name not in EVENTS or not _integer(event["seq"], 1) or not _integer(event["elapsed_ms"]):
        raise ProtocolError("invalid event name or sequence")
    timestamp = event["emitted_at"]
    try:
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise ValueError()
        datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise ProtocolError("invalid UTC timestamp") from exc
    for field in ("work_id", "track_id"):
        if event[field] is not None:
            uuid_text(event[field])
    track = event["track_id"]
    if track is None:
        if event["track_revision"] is not None:
            raise ProtocolError("revision without Track")
    elif event["work_id"] is None or not _integer(event["track_revision"], 1):
        raise ProtocolError("invalid Track envelope")
    if (name.startswith("track_") or name in {"stage_changed", "asr_progress"}) != (track is not None):
        raise ProtocolError("invalid event scope")
    if name == "work_discovered" and event["work_id"] is None:
        raise ProtocolError("missing Work")
    if event["stage"] is not None and event["stage"] not in STAGES:
        raise ProtocolError("unknown stage")
    if event["status"] is not None and event["status"] not in TERMINAL | {"ready", "running", "queued", "completed", "completed_with_errors", "cancelling", "interrupted", "protocol_failed"}:
        raise ProtocolError("unknown status")
    payload = event["payload"]
    if not isinstance(payload, dict) or payload.keys() - PAYLOAD_FIELDS[name]:
        raise ProtocolError("unsupported event payload")
    if not REQUIRED_PAYLOAD[name] <= payload.keys():
        raise ProtocolError("missing event payload fields")
    expected_status = {"track_queued": "queued", "track_started": "running", "stage_changed": "running",
                       "asr_progress": "running", "track_failed": "failed", "track_cancelled": "cancelled",
                       "track_not_started": "not_started"}.get(name)
    if expected_status and event["status"] != expected_status:
        raise ProtocolError("event status mismatch")
    if name == "track_finished" and (event["status"] not in {"success", "skipped", "copied", "transcribed"} or payload.get("outcome") != event["status"]):
        raise ProtocolError("terminal status mismatch")
    if name == "stage_changed" and (event["stage"] is None or payload.get("progress_mode") not in {"reported", "indeterminate"}):
        raise ProtocolError("invalid stage progress mode")
    if name == "track_failed":
        error = payload["error"]
        if not isinstance(error, dict) or set(error) != {"code", "retryable", "detail_ref"} or not isinstance(error["retryable"], bool):
            raise ProtocolError("invalid failure payload")
    if name in {"track_cancelled", "track_not_started"} and payload["reason"] not in {"dependency_failed", "run_cancelled", "run_aborted"}:
        raise ProtocolError("invalid interruption reason")
    if name == "control_ack":
        uuid_text(payload["request_id"])
        if payload["command"] not in {"cancel", "request_snapshot"} or not isinstance(payload["accepted"], bool):
            raise ProtocolError("invalid control acknowledgment")
    if name == "asr_progress":
        values = [payload.get(k) for k in ("reported_ratio", "current_audio_s", "total_audio_s")]
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in values) or values[2] <= 0 or not 0 <= values[0] <= 1 or payload.get("source") != "xxl_segment_start":
            raise ProtocolError("invalid ASR progress")
    return event


class JSONLDecoder:
    """Frame bytes before UTF-8 decoding; tolerate arbitrary read boundaries."""

    def __init__(self, run_id: str):
        self.run_id = uuid_text(run_id)
        self.buffer = bytearray()
        self.last_seq = 0
        self.revisions: dict[str, int] = {}
        self.seen: dict[int, dict] = {}
        self.terminal: set[str] = set()
        self.started: set[str] = set()

    def accept(self, event: dict) -> bool:
        validate_event(event, self.run_id)
        seq = event["seq"]
        if seq in self.seen:
            if self.seen[seq] != event:
                raise ProtocolError("conflicting duplicate event")
            return False
        if seq <= self.last_seq:
            raise ProtocolError("stale event sequence")
        if seq != self.last_seq + 1:
            raise ReconciliationRequired("event sequence gap")
        track = event["track_id"]
        if track:
            revision = event["track_revision"]
            previous = self.revisions.get(track, 0)
            if revision <= previous or track in self.terminal:
                raise ProtocolError("stale Track revision or late terminal event")
            if revision != previous + 1:
                raise ReconciliationRequired("Track revision gap")
            name = event["event"]
            if previous == 0 and name != "track_queued" or previous > 0 and name == "track_queued":
                raise ProtocolError("Track queue order mismatch")
            if name == "track_started":
                if track in self.started:
                    raise ProtocolError("Track started twice")
                self.started.add(track)
            if name in {"stage_changed", "asr_progress"} and track not in self.started:
                raise ProtocolError("Track progress before start")
            self.revisions[track] = revision
            if event["event"] in {"track_finished", "track_failed", "track_cancelled", "track_not_started"}:
                self.terminal.add(track)
        self.seen[seq] = event
        self.last_seq = seq
        return True

    def feed(self, data: bytes) -> list[dict]:
        if not isinstance(data, bytes):
            raise ProtocolError("JSONL input must be bytes")
        # Process complete frames incrementally; a large read of small frames is legal.
        output = []
        for fragment in data.splitlines(keepends=True):
            self.buffer.extend(fragment)
            if len(self.buffer) > MAX_LINE:
                raise ProtocolError("JSONL frame exceeds 64 KiB")
            if not self.buffer.endswith(b"\n"):
                continue
            frame = bytes(self.buffer[:-1])
            self.buffer.clear()
            try:
                event = json.loads(frame.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            except (ValueError, UnicodeError) as exc:
                raise ProtocolError("malformed JSONL frame") from exc
            if self.accept(event):
                output.append(event)
        return output

    def finish(self) -> None:
        if self.buffer:
            raise ProtocolError("truncated JSONL frame")
