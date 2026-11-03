"""GUI-thread state projected from frozen plans and authoritative run records."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time

from PyQt6.QtCore import QObject, QThread, pyqtSignal

TERMINAL = frozenset({"success", "skipped", "copied", "transcribed", "failed", "cancelled", "not_started"})
RESOLVED = TERMINAL - {"not_started"}
STAGE_LABELS = {
    "planning": "規劃中", "preflight": "前置檢查", "waiting_dependency": "等待代表音軌",
    "waiting_asr": "等待 ASR 資源", "asr": "ASR 轉錄中", "transcript_correction": "轉錄文字修正",
    "translation": "翻譯中（Sakura）", "opencc": "繁簡轉換", "traditional_correction": "繁體文字修正",
    "finalize": "寫入字幕", "copy": "複製字幕",
}
STATUS_LABELS = {
    "ready": "待執行", "queued": "排隊中", "running": "處理中", "success": "已完成翻譯",
    "transcribed": "已完成轉錄", "skipped": "已略過（既有輸出）", "copied": "已複製字幕",
    "failed": "失敗", "cancelled": "已取消", "not_started": "未執行", "excluded": "未納入本次執行",
    "empty": "無音軌", "finished": "已完成", "interrupted": "批次中斷", "all_skipped": "全數略過（既有輸出）",
}


def status_label(state) -> str:
    if isinstance(state, str):
        return STATUS_LABELS.get(state, STAGE_LABELS.get(state, state))
    if state.status == "running":
        label = STAGE_LABELS.get(state.stage, STATUS_LABELS["running"])
        return ("重用轉錄 · " if state.reused else "") + label
    label = STATUS_LABELS.get(state.status, state.status)
    if state.status == "not_started":
        label += "（依賴失敗）" if state.reason == "dependency_failed" else "（批次中斷）"
    return label


def format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


@dataclass
class TrackState:
    status: str = "ready"
    stage: str | None = None
    ratio: float | None = None
    started_ms: int | None = None
    ended_ms: int | None = None
    elapsed: float | None = None
    warnings: tuple[str, ...] = ()
    reused: bool = False
    reason: str | None = None
    revision: int = 0
    timing_unknown: bool = False


class RunModel(QObject):
    """One presentation authority; it never schedules work or examines outputs.

    Counters are updated per changed Track, so painting and summary polling do
    not scan the catalog. Timings are worker monotonic offsets, extrapolated
    locally only while a Track is known to be running.
    """

    changed = pyqtSignal(object)
    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plan = None
        self.run_id = None
        self.lifecycle = "IDLE"
        self._locked = False
        self.selected_work_ids: set[str] = set()
        self.tracks: dict[str, TrackState] = {}
        self.work_index = {}
        self.track_index = {}
        self.active_track_ids: set[str] = set()
        self.active_work_ids: set[str] = set()
        self._planned_ids: set[str] = set()
        self._counts = Counter()
        self._work_counts = {}
        self._work_start = {}
        self._work_end = {}
        self._work_warnings = Counter()
        self._work_unknown_ends = Counter()
        self._worker_ms = 0
        self._clock = time.monotonic()

    def _assert_thread(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("RunModel must be mutated on its GUI thread")

    @property
    def locked(self):
        return self._locked

    @locked.setter
    def locked(self, value):
        self._assert_thread()
        if self._locked != bool(value):
            self._locked = bool(value)
            self.changed.emit({"lock": True})

    def load_plan(self, plan):
        self._assert_thread()
        self.plan, self.run_id = plan, None
        self.work_index = {w.work_id: w for w in plan.works} if plan else {}
        self.track_index = {t.track_id: t for t in plan.tracks} if plan else {}
        self.selected_work_ids = ({w for w in plan.selected_work_ids if self.work_index[w].track_ids}
                                  if plan else set())
        excluded = set(plan.excluded_track_ids) if plan else set()
        self._planned_ids = set(self.track_index) - excluded
        self.tracks = {key: TrackState(status="excluded" if key in excluded else "ready",
                       warnings=("unverified_legacy",) if t.verification == "unverified_legacy" else (),
                       reason=t.task.reason) for key, t in self.track_index.items()}
        self._work_counts = {w: Counter() for w in self.work_index}
        self._work_start, self._work_end = {}, {}
        self._work_warnings = Counter()
        self._work_unknown_ends = Counter()
        self._counts = Counter()
        self.active_track_ids.clear()
        self.active_work_ids.clear()
        for key in self._planned_ids:
            work_id = self.track_index[key].work_id
            self._work_counts[work_id]["planned"] += 1
            self._work_counts[work_id]["ready"] += 1
            self._work_warnings[work_id] += bool(self.tracks[key].warnings)
        for work_id in self.selected_work_ids:
            self._counts.update(self._work_counts[work_id])
        self._worker_ms, self._clock = 0, time.monotonic()
        self.changed.emit({"reset": True})

    def begin_run(self, plan, run_id):
        self.load_plan(plan)
        self.run_id = run_id
        self.locked = True

    def set_work_selected(self, work_id, selected) -> bool:
        self._assert_thread()
        if self.locked or work_id not in self.work_index or not self.work_index[work_id].track_ids:
            return False
        selected = bool(selected)
        if (work_id in self.selected_work_ids) == selected:
            return True
        if selected:
            self.selected_work_ids.add(work_id)
            self._counts.update(self._work_counts[work_id])
        else:
            self.selected_work_ids.remove(work_id)
            self._counts.subtract(self._work_counts[work_id])
        self.changed.emit({"work_ids": (work_id,)})
        self.selection_changed.emit()
        return True

    def _now_ms(self):
        return self._worker_ms + int((time.monotonic() - self._clock) * 1000)

    def track_elapsed(self, track_id):
        state = self.tracks[track_id]
        if state.started_ms is None or state.timing_unknown:
            return None
        if state.ended_ms is not None:
            return max(0, state.ended_ms - state.started_ms) / 1000
        return max(0, self._now_ms() - state.started_ms) / 1000

    def _set_status(self, track_id, status):
        state, work_id = self.tracks[track_id], self.track_index[track_id].work_id
        old = state.status
        if status == old:
            return
        state.status = status
        if track_id not in self._planned_ids:
            return
        counts = self._work_counts[work_id]
        counts[old] -= 1
        counts[status] += 1
        if work_id in self.selected_work_ids:
            self._counts[old] -= 1
            self._counts[status] += 1
        if status == "running":
            self.active_track_ids.add(track_id)
        else:
            self.active_track_ids.discard(track_id)
        if counts["running"]:
            self.active_work_ids.add(work_id)
        else:
            self.active_work_ids.discard(work_id)

    def _notify(self, track_ids):
        if track_ids:
            self.changed.emit({"track_ids": tuple(track_ids),
                               "work_ids": tuple({self.track_index[t].work_id for t in track_ids})})

    def _merge_warnings(self, track_id, warnings):
        state = self.tracks[track_id]
        previous = bool(state.warnings)
        state.warnings = tuple(dict.fromkeys((*state.warnings, *warnings)))
        if track_id in self._planned_ids:
            self._work_warnings[self.track_index[track_id].work_id] += bool(state.warnings) - previous

    def _set_timing_unknown(self, track_id, unknown):
        state = self.tracks[track_id]
        if state.timing_unknown != unknown:
            self._work_unknown_ends[self.track_index[track_id].work_id] += int(unknown) - int(state.timing_unknown)
            state.timing_unknown = unknown

    def apply_event(self, event):
        self._assert_thread()
        if self.run_id is None or event.get("run_id") != self.run_id:
            return
        elapsed_ms = event.get("elapsed_ms", self._worker_ms)
        track_id = event.get("track_id")
        if track_id is None:
            if elapsed_ms >= self._worker_ms:
                self._worker_ms, self._clock = elapsed_ms, time.monotonic()
            return
        if (track_id not in self.tracks or track_id not in self._planned_ids or
                self.track_index[track_id].work_id not in self.selected_work_ids or
                event.get("work_id") != self.track_index[track_id].work_id):
            return
        state = self.tracks[track_id]
        revision = event.get("track_revision", 0)
        if revision <= state.revision or state.status in TERMINAL:
            return
        if elapsed_ms >= self._worker_ms:
            self._worker_ms, self._clock = elapsed_ms, time.monotonic()
        state.revision = revision
        name, payload = event["event"], event.get("payload", {})
        work_id = self.track_index[track_id].work_id
        self._set_status(track_id, event.get("status") or state.status)
        if name == "track_started":
            state.started_ms = elapsed_ms
            self._work_start[work_id] = min(self._work_start.get(work_id, elapsed_ms), elapsed_ms)
        stage = event.get("stage")
        if stage != state.stage:
            state.ratio = None
        state.stage = stage
        if name == "asr_progress" and stage == "asr":
            state.ratio = payload["reported_ratio"]
        if "reused" in payload:
            state.reused = bool(payload["reused"])
        if payload.get("warnings"):
            self._merge_warnings(track_id, payload["warnings"])
        if "reason" in payload:
            state.reason = payload["reason"]
        if name == "track_failed":
            state.reason = payload.get("error", {}).get("code", "TRACK_EXECUTION_FAILED")
        if state.status in TERMINAL:
            state.ratio = None
            state.ended_ms = elapsed_ms
            self._set_timing_unknown(track_id, False)
            self._work_end[work_id] = max(self._work_end.get(work_id, elapsed_ms), elapsed_ms)
        state.elapsed = self.track_elapsed(track_id)
        self._notify((track_id,))

    def _apply_records(self, record):
        if (self.run_id is None or record.get("run_id") != self.run_id or
                record.get("plan_hash") != self.plan.plan_hash):
            return
        changed = []
        for track_id, stored in record.get("tracks", {}).items():
            if track_id not in self.tracks or track_id not in self._planned_ids:
                continue
            state = self.tracks[track_id]
            revision = stored.get("revision", 0)
            if revision < state.revision:
                continue
            if stored.get("work_id", self.track_index[track_id].work_id) != self.track_index[track_id].work_id:
                continue
            state.revision = revision
            self._set_status(track_id, stored.get("status") or "ready")
            state.stage = stored.get("stage")
            # Snapshots do not persist native reports. Never retain an old ratio
            # across recovery and present it as current progress.
            state.ratio = None
            if "warnings" in stored:
                self._merge_warnings(track_id, stored["warnings"])
            state.reason = stored.get("reason", state.reason)
            work_id = self.track_index[track_id].work_id
            if stored.get("started_ms") is not None:
                state.started_ms = stored["started_ms"]
                self._work_start[work_id] = min(self._work_start.get(work_id, state.started_ms), state.started_ms)
            if stored.get("ended_ms") is not None:
                state.ended_ms = stored["ended_ms"]
            if state.status in TERMINAL:
                self._set_timing_unknown(track_id, state.ended_ms is None)
                if state.ended_ms is not None:
                    self._work_end[work_id] = max(self._work_end.get(work_id, 0), state.ended_ms)
            state.elapsed = self.track_elapsed(track_id)
            changed.append(track_id)
        self._notify(changed)

    def apply_snapshot(self, snapshot):
        self._assert_thread()
        self._apply_records(snapshot)

    def apply_reconciliation(self, result):
        self._assert_thread()
        self._apply_records(result)
        if (self.run_id is None or result.get("run_id") != self.run_id or
                result.get("plan_hash") != self.plan.plan_hash):
            return
        missing = []
        for track_id in result.get("missing_track_ids", ()):
            if track_id not in self._planned_ids or track_id not in self.tracks:
                continue
            self._set_status(track_id, "interrupted")
            state = self.tracks[track_id]
            state.stage, state.ratio = None, None
            state.reason = "UNCONFIRMED_RESULT"
            self._set_timing_unknown(track_id, True)
            state.elapsed = None
            missing.append(track_id)
        self._notify(missing)

    def summary(self) -> dict:
        counts = {status: self._counts[status] for status in TERMINAL}
        return {**counts, "total_works": len(self.work_index), "selected_works": len(self.selected_work_ids),
                "planned": self._counts["planned"], "resolved": sum(counts[s] for s in RESOLVED),
                "active_works": len(self.active_work_ids & self.selected_work_ids)}

    def work_summary(self, work_id) -> dict:
        counts = self._work_counts.get(work_id, Counter())
        planned = counts["planned"]
        resolved = sum(counts[s] for s in RESOLVED)
        terminal = resolved + counts["not_started"]
        if not planned:
            status, tone = "empty", "muted"
        elif counts["failed"]:
            status, tone = "failed", "danger"
        elif counts["running"]:
            status, tone = "running", "active"
        elif counts["cancelled"] or counts["not_started"] or counts["interrupted"]:
            status, tone = "interrupted", "warning"
        elif terminal == planned:
            status = ("all_skipped" if counts["skipped"] == planned else
                      "copied" if counts["copied"] == planned else
                      "transcribed" if counts["transcribed"] == planned else "finished")
            tone = "warning" if self._work_warnings[work_id] else "success"
        else:
            status, tone = "ready", "muted"
        start = self._work_start.get(work_id)
        end = self._work_end.get(work_id) if terminal == planned else self._now_ms()
        if not counts["running"] and self._work_unknown_ends[work_id]:
            end = None
        elapsed = None if start is None or end is None else max(0, end - start) / 1000
        return {"planned": planned, "resolved": resolved, "status": status, "tone": tone,
                "elapsed": elapsed, "progress": {"mode": "aggregate", "ratio": resolved / planned if planned else None,
                "caption": f"{resolved} / {planned}"}, "finished": bool(planned and terminal == planned),
                "failed": counts["failed"], "active": bool(counts["running"])}
