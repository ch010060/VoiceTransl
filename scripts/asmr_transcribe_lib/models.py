from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioTask:
    work_name: str
    work_dir: Path
    audio_path: Path
    root_relative_audio_path: Path
    relative_audio_path: Path
    final_srt_path: Path
    transcribe_srt_path: Path
    translate_srt_path: Path
    opencc_srt_path: Path
    status: str = "queued"
    action: str = "process"
    reason: str | None = None
    existing_subtitle_path: Path | None = None
