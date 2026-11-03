"""Bounded asynchronous curated GUI diagnostics; event authority stays in RunStore."""
from datetime import datetime, timezone
from pathlib import Path
import queue
import threading


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
                # Diagnostics cannot weaken cleanup or manufacture successful work.
                self.dropped += 1

    def close(self):
        self.closed = True
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            # Existing records still drain; daemon exit never blocks Qt shutdown.
            pass
