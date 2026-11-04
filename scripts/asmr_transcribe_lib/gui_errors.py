"""Curated execution failures; never retain exception messages or subtitle text."""
from pathlib import Path
import re
from .srt_normalization import TimestampOrderError

NORMALIZATION_REJECTIONS = {
    "empty", "index", "timestamp_range", "format", "end_before_start",
    "displacement_limit", "repair_exceeds_end", "count_limit", "ratio_limit",
    "serialization", "strict_validation", "audit_store_required",
}

ERROR_REASONS = {
    "ASR_PROCESS_FAILED": "語音辨識程序未正常完成",
    "ASR_EXECUTION_FAILED": "語音辨識呼叫失敗",
    "ASR_OUTPUT_MISSING": "語音辨識未產生預期輸出",
    "SOURCE_CHANGED": "來源音訊與計畫不符",
    "OUTPUT_CHANGED": "輸出與計畫不符",
    "SRT_TIMESTAMP_ORDER_INVALID": "字幕時間戳順序不合法；未自動修復",
    "SRT_VALIDATION_FAILED": "字幕格式或時間軸驗證失敗",
    "TRANSLATION_FAILED": "翻譯呼叫失敗",
    "OPENCC_FAILED": "繁簡轉換失敗",
    "OUTPUT_PUBLICATION_FAILED": "字幕輸出發布失敗",
    "TRANSCRIPT_CORRECTION_FAILED": "轉錄文字修正失敗",
    "DEPENDENCY_FAILED": "必要的上游音軌未成功完成",
    "CANCELLATION_REQUESTED": "使用者要求停止",
    "RUN_ABORTED": "執行中斷，並非使用者停止",
    "INTERNAL_EXECUTION_ERROR": "音軌執行發生內部錯誤",
    "TRACK_EXECUTION_FAILED": "舊紀錄未保留詳細錯誤分類",
}


class ExecutionFailure(Exception):
    def __init__(self, code, cause=None, details=None):
        self.code = code
        self.exception_type = type(cause).__name__ if cause else None
        self.details = details or {}
        super().__init__(code)


def failure_record(code, stage, *, exception_type=None, details=None):
    code = code if code in ERROR_REASONS else "INTERNAL_EXECUTION_ERROR"
    return {"code": code, "stage": stage, "reason": ERROR_REASONS[code],
            "exception_type": exception_type, "details": details or {}}


def curate_failure(value, stage):
    value = value if isinstance(value, dict) else {}
    kind = value.get("exception_type")
    if not isinstance(kind, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,79}", kind):
        kind = None
    details = {}
    raw = value.get("details", {})
    if isinstance(raw, dict) and isinstance(raw.get("anomaly_count"), int):
        examples = []
        timing = r"\d{2,4}:\d{2}:\d{2},\d{3} --> \d{2,4}:\d{2}:\d{2},\d{3}"
        for example in raw.get("examples", [])[:5]:
            if (isinstance(example, dict) and isinstance(example.get("segment"), int)
                    and isinstance(example.get("backward_ms"), int)
                    and re.fullmatch(timing, str(example.get("timestamp", "")))
                    and (example.get("previous_timestamp") is None or re.fullmatch(timing, str(example["previous_timestamp"])))):
                examples.append({k: example[k] for k in ("segment", "timestamp", "previous_timestamp", "backward_ms")})
        details = {"anomaly_count": raw["anomaly_count"], "examples": examples, "automatic_repair": "DEFERRED"}
    if isinstance(raw, dict) and raw.get("normalization_rejection") in NORMALIZATION_REJECTIONS:
        details.update(normalization_rejection=raw["normalization_rejection"], automatic_repair="REJECTED")
    return failure_record(value.get("code", "TRACK_EXECUTION_FAILED"), stage, exception_type=kind, details=details)


def srt_failure(error, path):
    """On failure only, retain numeric timing evidence without any caption text."""
    code = "SRT_VALIDATION_FAILED"
    details = {}
    rejected = isinstance(error, TimestampOrderError)
    if error.args == ("SRT timestamps out of order",) or rejected and error.category in {
            "end_before_start", "displacement_limit", "repair_exceeds_end", "count_limit", "ratio_limit"}:
        code = "SRT_TIMESTAMP_ORDER_INVALID"
        from .srt_utils import parse_srt_file
        previous = None
        defects = []
        def milliseconds(value):
            h, m, tail = value.split(":")
            s, ms = tail.split(",")
            return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)
        for position, block in enumerate(parse_srt_file(Path(path)), 1):
            left, right = map(milliseconds, block.timestamp.split(" --> "))
            if right < left or previous is not None and left < previous[0]:
                defects.append({"segment": position, "timestamp": block.timestamp,
                                "previous_timestamp": previous[1] if previous else None,
                                "backward_ms": max(0, previous[0] - left) if previous else 0})
            previous = (left, block.timestamp)
        details = {"anomaly_count": len(defects), "examples": defects[:5], "automatic_repair": "DEFERRED"}
    if rejected:
        code = "SRT_TIMESTAMP_ORDER_INVALID"
        details.update(normalization_rejection=error.category, automatic_repair="REJECTED")
    return ExecutionFailure(code, error, details)
