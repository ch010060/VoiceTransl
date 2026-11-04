"""Pure presentation over frozen identities; receipts/journals remain authority."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import queue
import re
import threading

from .gui_errors import ERROR_REASONS
from .srt_normalization import WARNING_CODE


class DiagnosticLog:
    def __init__(self, path):
        self.path = Path(path)
        self.queue = queue.Queue(maxsize=2048)
        self.dropped = 0
        self.closed = False
        self.thread = threading.Thread(target=self._write, name="asmr-diagnostics", daemon=True)
        self.thread.start()

    def append(self, root, message, context=""):
        if self.closed:
            return
        line = str(message).replace("\r", " ").replace("\n", " ")[:1024]
        try:
            self.queue.put_nowait((root, line, context))
        except queue.Full:
            self.dropped += 1

    def _write(self):
        while True:
            try:
                item = self.queue.get(timeout=.25)
            except queue.Empty:
                if self.closed:
                    return
                continue
            if item is None:
                return
            root, message, context = item
            try:
                if not root or self.path.resolve().is_relative_to(Path(root).resolve()):
                    continue
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as output:
                    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                    if self.dropped:
                        output.write(f"{now} DIAGNOSTIC_ENTRIES_DROPPED {self.dropped}\n")
                        self.dropped = 0
                    output.write(f"{now} {context} {message}\n")
            except OSError:
                self.dropped += 1

    def close(self, wait=False):
        self.closed = True
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if wait:
            self.thread.join(timeout=2)


def _text(value):
    return "".join(c if c.isprintable() else "?" for c in str(value))[:512]


def _duration(milliseconds):
    if milliseconds is None:
        return "時間未確認"
    seconds = max(0, milliseconds // 1000)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"


class RunLogFormatter:
    def __init__(self, plan, run_id, run_dir=None):
        self.plan, self.run_id = plan, run_id
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.tracks = {t.track_id: t for t in plan.planned_tracks}
        self.works = {w.work_id: w for w in plan.works}
        self.labels = {}
        for work in plan.works:
            match = re.search(r"RJ\d+", work.relative_path, re.IGNORECASE)
            self.labels[work.work_id] = match.group(0).upper() if match else _text(work.relative_path or "未分類作品")
        self.paths = {key: _text(t.task.audio_path.relative_to(t.task.work_dir).as_posix()) for key, t in self.tracks.items()}
        self.revisions, self.started, self.stages, self.terminal = {}, {}, {}, {}
        self.work_counts = {key: Counter() for key in self.works}
        self.work_totals = Counter(t.work_id for t in self.tracks.values())
        self.work_announced, self.work_started = set(), set()

    def event(self, event):
        if event.get("run_id") != self.run_id:
            return []
        name = event["event"]
        if name in {"heartbeat", "asr_progress", "track_queued", "worker_ready", "work_discovered"}:
            return []
        try:
            clock = datetime.fromisoformat(event["emitted_at"].replace("Z", "+00:00")).astimezone().strftime("%H:%M:%S")
        except (KeyError, ValueError):
            clock = datetime.now().astimezone().strftime("%H:%M:%S")
        prefix = f"[{clock}]"
        key, payload = event.get("track_id"), event.get("payload", {})
        context = f"run={self.run_id} event={name}"
        if self.run_dir is not None:
            context += f" directory={self.run_dir}"
        if key is None:
            if name == "run_started":
                works = len({t.work_id for t in self.tracks.values()})
                return [(f"{prefix} [批次] 開始執行 · {works} 作品／{len(self.tracks)} 音軌", context)]
            return []
        track = self.tracks.get(key)
        if track is None or event.get("work_id") != track.work_id or key in self.terminal:
            return []
        if name in {"track_finished", "track_failed", "track_cancelled", "track_not_started"} and not payload.get("receipt_id"):
            return []
        revision = event.get("track_revision", 0)
        if revision <= self.revisions.get(key, 0):
            return []
        self.revisions[key] = revision
        context += f" work={track.work_id} track={key} audio={self.paths[key]} stage={event.get('stage') or self.stages.get(key)}"
        if event.get("recovered"):
            context += " recovered=True event_time=unconfirmed"
        if track.dependencies:
            source = track.task.existing_subtitle_path
            if source is not None:
                try:
                    source = source.relative_to(track.task.work_dir).as_posix()
                except ValueError:
                    source = "[來源位於作品外]"
            context += f" owner={track.dependencies[0]} copy_source={_text(source)}"
        lines = []
        if track.work_id not in self.work_started:
            self.work_started.add(track.work_id)
            lines.append((f"{prefix} [{self.labels[track.work_id]}] 開始處理 · {self.work_totals[track.work_id]} 音軌", context))
        title = f"{prefix} [{self.labels[track.work_id]}] [{self.paths[key]}]"
        elapsed = event.get("elapsed_ms", 0)
        if name == "track_started":
            self.started[key] = elapsed
            lines.append((title + (" 開始共用字幕" if track.shared_output_owner else " 開始複製字幕" if track.dependencies else " 開始處理"), context))
        elif name == "stage_changed":
            stage = event.get("stage")
            previous = self.stages.get(key)
            self.stages[key] = stage
            if stage == previous or stage in {"waiting_asr", "waiting_dependency", "copy", "finalize", "traditional_correction"}:
                return lines
            message = {"asr": "開始語音辨識", "transcript_correction": "語音辨識完成" if previous == "asr" else "重用既有轉錄",
                       "translation": "開始翻譯（Sakura）", "opencc": "翻譯完成，開始繁簡轉換"}.get(stage)
            if message:
                lines.append((title + " " + message, context))
        elif name in {"track_finished", "track_failed", "track_cancelled", "track_not_started"} and payload.get("receipt_id"):
            status = event["status"]
            self.terminal[key] = status
            self.work_counts[track.work_id][status] += 1
            duration = _duration(elapsed - self.started[key] if key in self.started else None)
            message = {"success": "完成翻譯", "transcribed": "完成轉錄", "copied": "共用字幕完成" if track.shared_output_owner else "複製完成",
                       "skipped": "略過既有字幕", "failed": "失敗", "cancelled": "已停止", "not_started": "未執行"}[status]
            if status == "failed":
                code = payload.get("error", {}).get("code")
                code = code if code in ERROR_REASONS else "TRACK_EXECUTION_FAILED"
                message += " · " + code + " · " + ERROR_REASONS[code]
                reference = payload.get("error", {}).get("detail_ref", "")
                if re.fullmatch(r"(?:logs|receipts)/[0-9a-f-]+(?:\.error)?\.json", str(reference)):
                    context += " diagnostic=" + reference
            elif status in {"cancelled", "not_started"}:
                message += " · " + {"dependency_failed": "依賴失敗", "run_cancelled": "使用者停止", "run_aborted": "執行中斷"}.get(payload.get("reason"), "原因待核對")
            if WARNING_CODE in payload.get("warnings", []):
                message += " · " + WARNING_CODE + "（時間軸已受限正規化，證據見收據）"
            context += f" status={status} receipt={payload['receipt_id']} duration={duration}"
            lines.append((f"{title} {message} · {duration}", context))
            lines.extend(self._work_summary(track.work_id, prefix, context))
        return lines

    def _work_summary(self, work_id, prefix, context):
        counts = self.work_counts[work_id]
        if work_id in self.work_announced or sum(counts.values()) != self.work_totals[work_id]:
            return []
        self.work_announced.add(work_id)
        state = "（含失敗）" if counts["failed"] else "（未完整完成）" if counts["cancelled"] or counts["not_started"] else ""
        summary = "／".join(f"{label} {counts[key]}" for key, label in
                            [("success", "成功"), ("transcribed", "僅轉錄"), ("copied", "複製"), ("skipped", "略過"),
                             ("failed", "失敗"), ("cancelled", "停止"), ("not_started", "未執行")])
        return [(f"{prefix} [{self.labels[work_id]}] 作品結算{state} · {summary}", context)]

    def reconcile(self, result, cleanup, *, user_stop=False):
        if result.get("run_id") != self.run_id or result.get("plan_hash") != self.plan.plan_hash:
            return []
        lines = []
        for key, stored in result.get("tracks", {}).items():
            if key not in self.tracks or key in self.terminal:
                continue
            status = stored.get("status")
            if status not in {"success", "transcribed", "copied", "skipped", "failed", "cancelled", "not_started"} or not stored.get("receipt_ref"):
                continue
            if stored.get("started_ms") is not None and stored.get("ended_ms") is not None:
                self.started[key] = stored["started_ms"]
            else:
                self.started.pop(key, None)
            error = stored.get("error") or {}
            code = error.get("code", stored.get("reason", "TRACK_EXECUTION_FAILED"))
            lines.extend(self.event({"run_id": self.run_id, "track_id": key, "work_id": self.tracks[key].work_id,
                "track_revision": max(stored.get("revision", 0), self.revisions.get(key, 0) + 1),
                "event": {"failed": "track_failed", "cancelled": "track_cancelled", "not_started": "track_not_started"}.get(status, "track_finished"),
                "stage": error.get("stage"), "status": status, "elapsed_ms": stored.get("ended_ms") or 0, "recovered": True,
                "payload": {"receipt_id": stored["receipt_ref"], "reason": stored.get("reason"),
                            "warnings": stored.get("warnings", []),
                            "error": {"code": code, "detail_ref": error.get("diagnostic_ref", stored["receipt_ref"])}}}))
        outcome = result.get("outcome", "interrupted")
        reason = ("使用者要求停止" if user_stop or outcome == "cancelled" else
                  "通訊協定失敗" if outcome == "protocol_failed" else
                  "執行未完整結算／Worker 異常退出" if outcome == "interrupted" or not result.get("complete") else
                  "含音軌失敗" if outcome == "completed_with_errors" else "執行完成")
        counts = result.get("counts", {})
        summary = "／".join(f"{label} {counts.get(key, 0)}" for key, label in
                            [("success", "成功"), ("transcribed", "僅轉錄"), ("copied", "複製"), ("skipped", "略過"),
                             ("failed", "失敗"), ("cancelled", "停止"), ("not_started", "未執行")])
        context = f"run={self.run_id} outcome={outcome} complete={result.get('complete')}"
        location = str(self.run_dir / "logs") if self.run_dir is not None else "完整日誌按鈕"
        lines.append((f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] [批次] {reason} · 計畫 {len(self.tracks)} 音軌 · {summary}"
                      f" · cleanup={cleanup.get('cleanup_verified')} active={cleanup.get('active_count')}"
                      f" · 完整日誌／診斷：{location}", context))
        return lines
