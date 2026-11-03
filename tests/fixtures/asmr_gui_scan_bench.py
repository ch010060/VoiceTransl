"""Opt-in read-only scan benchmark through the real Controller and Windows Job.

All scan artifacts go outside the library. No ASR, translation, GUI debug server,
or persistent cache. Profiling wrappers exist only in this benchmark process.
"""
from __future__ import annotations

from collections import Counter
import argparse
import cProfile
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

ENTRY = time.perf_counter()
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))


class Metrics:
    def __init__(self):
        self.calls, self.inclusive, self.exclusive = Counter(), Counter(), Counter()
        self.local = threading.local()
        self.lock = threading.Lock()
        self.first = {}

    def wrap(self, owner, name, label=None, observer=None):
        original = getattr(owner, name)
        label = label or name

        def measured(*args, **kwargs):
            stack = getattr(self.local, "stack", None)
            if stack is None:
                self.local.stack = stack = []
            frame = [time.perf_counter(), 0.0]
            stack.append(frame)
            try:
                if observer:
                    observer(*args, **kwargs)
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - frame[0]
                stack.pop()
                if stack:
                    stack[-1][1] += elapsed
                with self.lock:
                    self.calls[label] += 1
                    self.inclusive[label] += elapsed
                    self.exclusive[label] += elapsed - frame[1]
                    self.first.setdefault(label, frame[0])

        setattr(owner, name, measured)

    def result(self):
        return {key: {"calls": self.calls[key], "inclusive_ms": round(self.inclusive[key] * 1000, 3),
                      "exclusive_ms": round(self.exclusive[key] * 1000, 3)} for key in self.calls}


def instrument(metrics, detail=False):
    from scripts.asmr_transcribe_lib import planner, gui_plan, gui_plan_io
    groups = []
    def grouping(tasks):
        try:
            from scripts.asmr_transcribe_lib.scan_cache import current_phase
            phase = current_phase()
        except ImportError:
            phase = None  # Allows the same harness on the immutable baseline.
        if phase is None or tuple(task.audio_path for task in tasks) not in phase.groups:
            groups.append(len(tasks))
    metrics.wrap(planner, "_group_same_content_tasks", "duplicate_matching", grouping)
    gui_plan._group_same_content_tasks = planner._group_same_content_tasks
    metrics.wrap(planner, "discover_audio_tasks", "audio_discovery")
    gui_plan.discover_audio_tasks = planner.discover_audio_tasks
    for module, names in ((gui_plan, ("_existing_common_subtitle", "file_stamp", "_namespace_issues", "build_execution_plan")),
                          (gui_plan_io, ("_validate", "_audio_membership", "encode_plan", "decode_plan", "save_plan", "load_plan"))):
        for name in names:
            metrics.wrap(module, name)
    if detail:
        for name in ("resolve", "stat", "lstat", "exists", "is_file", "read_bytes"):
            metrics.wrap(Path, name, "filesystem_" + name)
        metrics.wrap(os, "scandir", "directory_enumerations")
        metrics.subtitle_bytes = 0
        read_bytes = Path.read_bytes
        def read_subtitle(path):
            data = read_bytes(path)
            if path.suffix.lower() in {".srt", ".vtt", ".ass", ".ssa", ".lrc", ".tmp"}:
                metrics.subtitle_bytes += len(data)
            return data
        Path.read_bytes = read_subtitle
    return groups


def worker():
    detail = os.environ.get("VOICETRANSL_BENCH_DETAIL") == "1"
    metrics = Metrics()
    groups = instrument(metrics, detail)
    from scripts import asmr_gui_worker
    from scripts.asmr_transcribe_lib import gui_plan, gui_plan_io
    asmr_gui_worker.build_execution_plan = gui_plan.build_execution_plan
    asmr_gui_worker.save_plan = gui_plan_io.save_plan
    metrics.wrap(asmr_gui_worker, "assert_current_in_job", "worker_job_validation")
    args = sys.argv[2:]
    value = lambda key: args[args.index(key) + 1]
    destination = Path(value("--state-root")) / "bench" / (value("--run-id") + ".json")
    profile = cProfile.Profile() if os.environ.get("VOICETRANSL_BENCH_CPU") == "1" else None
    if profile:
        profile.enable()
    try:
        return asmr_gui_worker.main(args)
    finally:
        if profile:
            profile.disable()
            profile.dump_stats(str(destination.with_suffix(".pstats")))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"entry_at": ENTRY, "first_plan_at": metrics.first.get("build_execution_plan"),
            "metrics": metrics.result(), "group_sizes": groups,
            "pair_comparisons": sum(n * (n - 1) // 2 for n in groups),
            "subtitle_bytes_hashed": getattr(metrics, "subtitle_bytes", None)}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--cpu-profile", action="store_true")
    parser.add_argument("--history-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    base = Path(tempfile.gettempdir()) / "VoiceTransl" / "Tests" / ("scan-bench-" + uuid4().hex)
    if base.resolve().is_relative_to(root) or args.output.resolve().is_relative_to(root):
        parser.error("benchmark outputs must be outside the audio library")
    base.mkdir(parents=True)
    (base / "bench").mkdir()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["VOICETRANSL_BENCH_DETAIL"] = "1" if args.detail else "0"
    os.environ["VOICETRANSL_BENCH_CPU"] = "1" if args.cpu_profile else "0"
    from PyQt6.QtCore import QEventLoop, QObject, QSettings, QTimer, pyqtSignal
    from PyQt6.QtWidgets import QApplication
    from scripts.asmr_gui.controller import Controller
    from scripts.asmr_gui_runtime import ManagedOperation
    from scripts.asmr_gui import controller as controller_module, window as window_module
    from scripts.asmr_gui.run_model import RunModel
    from scripts.asmr_gui.tree_model import WorkTrackModel
    from scripts.asmr_transcribe_lib import gui_plan_io
    metrics = Metrics()
    instrument(metrics, args.detail)
    controller_module.load_plan = gui_plan_io.load_plan
    for cls, name in ((Controller, "_historical_display"), (RunModel, "load_plan"), (WorkTrackModel, "_rebuild")):
        metrics.wrap(cls, name, cls.__name__ + "." + name)
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    window_module.QSettings = lambda *unused: QSettings(str(base / "preferences.ini"), QSettings.Format.IniFormat)

    class QuietHealth(QObject):
        updated = pyqtSignal(dict)

        def __init__(self, **kwargs):
            super().__init__(kwargs.get("parent"))

        def configure(self, *unused):
            pass

        set_paused = configure
        close = configure

    managed_operations = []

    def operation_factory(operation, state_root, command_factory, **kwargs):
        def command(managed):
            argv = command_factory(managed)
            return [*argv[:3], str(Path(__file__).resolve()), "--profile-worker", *argv[4:]]
        managed = ManagedOperation(operation, state_root, command, **kwargs)
        managed_operations.append(managed)
        return managed

    runs = []
    for pair in range(args.warmup + args.pairs):
        controller = Controller(state_root=base, operation_factory=operation_factory, health_monitor_factory=QuietHealth)
        window = window_module.BatchWindow(controller)
        if args.history_root:
            view = SimpleNamespace(state_root=args.history_root, _history_file=args.history_root / "gui-last-result.json")
            controller._historical_display = lambda intent, plan: Controller._historical_display(view, intent, plan)
        try:
            for kind in ("cold", "rescan"):
                metrics.calls.clear()
                metrics.inclusive.clear()
                metrics.exclusive.clear()
                loop = QEventLoop()
                timer = QTimer()
                timer.setSingleShot(True)
                timer.timeout.connect(loop.quit)
                ready_at = []

                def state_changed(state):
                    if state in {"READY", "FINISHED", "ERROR", "CLEANUP_FAILED"}:
                        ready_at.append(time.perf_counter())
                        loop.quit()

                controller.state_changed.connect(state_changed)
                started = time.perf_counter()
                assert controller.scan(str(root))
                timer.start(180000)
                loop.exec()
                timer.stop()
                controller.state_changed.disconnect(state_changed)
                elapsed = time.perf_counter() - started
                if not ready_at or controller.state not in {"READY", "FINISHED"}:
                    raise RuntimeError("scan failed: " + controller.last_error)
                managed = managed_operations[-1]
                child = json.loads((base / "bench" / (managed.run_id + ".json")).read_text())
                plan = controller.run_model.plan
                record = {"pair": pair, "warmup": pair < args.warmup, "kind": kind,
                    "total_ms": round(elapsed * 1000, 3), "works": len(plan.works), "tracks": len(plan.tracks),
                    "max_tracks_per_work": max((len(work.track_ids) for work in plan.works), default=0),
                    "startup_to_worker_entry_ms": round((child["entry_at"] - started) * 1000, 3),
                    "startup_to_plan_ms": round((child["first_plan_at"] - started) * 1000, 3),
                    "parent_metrics": metrics.result(), "worker_metrics": child["metrics"],
                    "pair_comparisons": child["pair_comparisons"], "group_calls": len(child["group_sizes"]),
                    "plan_hash": plan.plan_hash, "plan_file": str(managed.run_dir / "plan.json"),
                    "cleanup_verified": controller.last_cleanup is None or managed._result.get("cleanup_verified"),
                    "run_dir": str(managed.run_dir)}
                runs.append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"base": str(base), "runs": runs}, indent=2), encoding="utf-8")
                print(json.dumps({key: record[key] for key in ("pair", "warmup", "kind", "total_ms", "works", "tracks")}), flush=True)
        finally:
            if controller.busy:
                controller.stop()
                deadline = time.perf_counter() + 10
                while controller.busy and time.perf_counter() < deadline:
                    app.processEvents()
            controller.request_close()
            window.close()
            window.deleteLater()
            app.processEvents()
    summary = {kind: {"median_ms": statistics.median([r["total_ms"] for r in runs if r["kind"] == kind and not r["warmup"]]),
        "worst_ms": max(r["total_ms"] for r in runs if r["kind"] == kind and not r["warmup"])} for kind in ("cold", "rescan")}
    args.output.write_text(json.dumps({"base": str(base), "runs": runs, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--profile-worker"]:
        raise SystemExit(worker())
    main()
