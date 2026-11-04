"""Optional in-process observation/checkpoints. No Qt, IPC or process-stop claims."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import threading


class PipelineCancelled(Exception):
    pass


class ObserverError(RuntimeError):
    """Observation failure is fatal to the observed run, not an audio failure."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = "run_cancelled"
        self._lock = threading.Lock()

    def cancel(self, reason="run_cancelled") -> None:
        with self._lock:
            if not self._event.is_set():
                self._reason = reason
                self._event.set()

    @property
    def reason(self):
        return self._reason

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def checkpoint(self) -> None:
        if self.cancelled:
            raise PipelineCancelled(self._reason)


@dataclass(frozen=True)
class PipelineEvent:
    event: str
    track_id: str
    stage: str | None = None
    status: str | None = None
    progress: tuple[float, float, float] | None = None
    reason: str | None = None
    failure: dict | None = None


@dataclass(frozen=True)
class TaskRuntime:
    stage: Callable[[str], None]
    checkpoint: Callable[[], None]
    validate: Callable[[Path], None]
    publish: Callable[[Path, Path], None]

    def enter(self, stage: str) -> None:
        self.checkpoint()
        self.stage(stage)
        self.checkpoint()
