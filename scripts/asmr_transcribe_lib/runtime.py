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

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def checkpoint(self) -> None:
        if self.cancelled:
            raise PipelineCancelled("run_cancelled")


@dataclass(frozen=True)
class PipelineEvent:
    event: str
    track_id: str
    stage: str | None = None
    status: str | None = None
    progress: tuple[float, float, float] | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = ()


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
