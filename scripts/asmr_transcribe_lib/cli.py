from __future__ import annotations

import argparse
from dataclasses import replace
import sys
import threading
from pathlib import Path
from time import monotonic

from .local_faster_whisper import (
    FasterWhisperConfig,
    find_faster_whisper_exe,
    find_faster_whisper_model_dir,
    preflight_faster_whisper,
    preflight_faster_whisper_model,
    run_faster_whisper_transcription,
)
from .kotoba_faster_whisper import (
    DEFAULT_KOTOBA_MODEL,
    KotobaFasterWhisperConfig,
    preflight_kotoba_faster_whisper,
    run_kotoba_faster_whisper_transcription,
)
from .models import AudioTask
from .openai_compatible import preflight_model
from .planner import discover_audio_tasks
from .pipeline import TaskResult, run_tasks
from .translation import translate_srt_to_simplified

DEFAULT_TRANSLATE_API = "http://127.0.0.1:8080/v1"

# Vocabulary-biasing prompt for adult ASMR Japanese.
# Covers confirmed-misrecognized terms (搾精/射精/精気/精液/おちんちん/サキュバス/シーツ/舌)
# plus common adult-ASMR vocabulary that is rare in standard ASR training data.
DEFAULT_ASMR_INITIAL_PROMPT = (
    "日本語の成人向けASMR音声作品です。"
    "サキュバスや魔法少女が耳舐め、耳かき、手コキ、フェラチオ、クンニで奉仕します。"
    "勃起したおちんちん、おちんぽを舌で舐め、搾精し射精させます。"
    "精液、愛液でシーツが汚れ、精気を吸い取ります。"
    "乳首、おっぱい、膣、亀頭、絶頂、中出し、潮吹き。"
    "気持ちいい、イく、出して。"
)
DEFAULT_SRC_LANG = "ja"
DEFAULT_TRANSCRIBE_MODEL = "faster-whisper-large-v3"
DEFAULT_TRANSLATE_MODEL = "sakura-14b-qwen2.5-v1.0-q4km.gguf"
DEFAULT_PARALLEL = 5
DEFAULT_TRANSCRIBE_PARALLEL = 1
DEFAULT_FASTER_WHISPER_PARAM = Path("whisper-faster") / "param.txt"
# Sentinel stored in the transcription-progress map once a task moves on to translation.
_TRANSLATING = (-1.0, 0.0, 0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asmr_transcribe_srt_batch",
        description="Batch scan ASMR audio folders for the transcription/translation SRT pipeline.",
    )
    parser.add_argument("--path", required=True, help="Root path containing first-level work folders.")
    parser.add_argument("--src_lang", default=DEFAULT_SRC_LANG)
    parser.add_argument("--transcribe_backend", choices=["xxl", "kotoba"], default="xxl", help="Transcription backend. 'xxl' keeps the project-local Faster-Whisper-XXL binary; 'kotoba' uses Python faster-whisper.")
    parser.add_argument("--transcribe_model", default=None)
    parser.add_argument("--transcribe_compute_type", default="float16")
    parser.add_argument("--kotoba_device", default="cuda", help="Device for the experimental Kotoba Python adapter; default uses CUDA.")
    parser.add_argument("--kotoba_chunk_length", type=int, default=15)
    parser.add_argument("--faster_whisper_param", default=str(DEFAULT_FASTER_WHISPER_PARAM), help="Project-local Faster Whisper command template file (display-only, does not alter the built command).")
    parser.add_argument("--fast_transcribe", action="store_true", help="Speed preset: batched inference (batch_size=16, --unmerged). ~3.6x faster but lower quality — do NOT use for production subtitle output.")
    parser.add_argument("--translate_chunk_size", type=int, default=20, help="Max SRT lines per translation request. 20 measured best for Sakura v1.0 (fewer line-count mismatches than 30 at equal or lower total time).")
    parser.add_argument("--initial_prompt", default=None, help="Custom Whisper decoder initial prompt. Overrides the built-in ASMR vocabulary prompt.")
    parser.add_argument("--no_asmr_prompt", action="store_true", help="Disable the built-in ASMR vocabulary prompt. Use when transcribing non-adult content.")
    parser.add_argument("--translate_model", default=DEFAULT_TRANSLATE_MODEL)
    parser.add_argument("--translate_api", default=DEFAULT_TRANSLATE_API)
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL, help="Overall pipeline workers. Transcription is separately gated by --transcribe_parallel.")
    parser.add_argument("--transcribe_parallel", type=int, default=DEFAULT_TRANSCRIBE_PARALLEL, help="Maximum concurrent transcription jobs. Default 1 avoids GPU contention for large-v3.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and print the queue without calling APIs or writing outputs.")
    parser.add_argument("--execute", action="store_true", help="Run preflight and process the discovered queue.")
    parser.add_argument("--transcribe_only", action="store_true", help="Only write *-transcribe.srt next to each audio file; skip translation and the translation API preflight.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of queued audio files to process in this run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.parallel < 1:
        parser.error("--parallel must be >= 1")
    if args.transcribe_parallel < 1:
        parser.error("--transcribe_parallel must be >= 1")
    if args.limit is not None and args.limit < 1:
        print("--limit must be >= 1", file=sys.stderr)
        return 2

    root = Path(args.path)
    project_root = Path(__file__).resolve().parents[2]
    effective_transcribe_model = _effective_transcribe_model(args.transcribe_backend, args.transcribe_model)
    faster_whisper_param = Path(args.faster_whisper_param)
    if not faster_whisper_param.is_absolute():
        faster_whisper_param = project_root / faster_whisper_param
    faster_whisper_exe = find_faster_whisper_exe(project_root)
    started = monotonic()
    tasks = discover_audio_tasks(root)

    if args.execute:
        print("Execute mode: enabled")
    else:
        print("Preflight skipped: dry-run mode")

    print(f"Transcription mode: {_transcription_mode_label(args.transcribe_backend)}")
    print(f"Faster-Whisper param file: {_display_project_path(faster_whisper_param, project_root)}")
    exe_display = _display_project_path(faster_whisper_exe, project_root) if faster_whisper_exe else "<missing>"
    print(f"Faster-Whisper executable: {exe_display}")
    print(f"Faster-Whisper command template (display only): {_safe_for_stdout(_truncate_text(_read_command_template(faster_whisper_param), 120))}")
    print(f"Transcription model: {effective_transcribe_model}")
    effective_prompt = _effective_initial_prompt(args.initial_prompt, args.no_asmr_prompt)
    prompt_display = _safe_for_stdout(_truncate_text(effective_prompt, 80)) if effective_prompt else "<disabled>"
    print(f"Initial prompt: {prompt_display}")
    if args.transcribe_backend == "kotoba":
        print(f"Kotoba device: {args.kotoba_device}")
        print(f"Kotoba chunk length: {args.kotoba_chunk_length}")
    print(f"Translation API: {args.translate_api}")
    print(f"Translation model: {args.translate_model}")
    print(f"Source language: {args.src_lang}")

    if not args.execute:
        print_progress_snapshot(tasks, args.parallel, monotonic() - started, args.transcribe_parallel)
        return 0

    try:
        if args.transcribe_only:
            print("Preflight translation API: skipped (--transcribe_only)")
        else:
            preflight_model(args.translate_api, args.translate_model)
            print("Preflight translation API: OK")
        transcriber = None
        if args.transcribe_backend == "kotoba":
            preflight_kotoba_faster_whisper()
            print("Preflight kotoba python faster-whisper: OK")
            config = KotobaFasterWhisperConfig(
                model=effective_transcribe_model,
                language=args.src_lang,
                device=args.kotoba_device,
                compute_type=args.transcribe_compute_type,
                chunk_length=args.kotoba_chunk_length,
                condition_on_previous_text=False,
            )
            transcriber = run_kotoba_faster_whisper_transcription
        else:
            faster_whisper_exe = preflight_faster_whisper(project_root)
            print(f"Preflight faster-whisper: {_display_project_path(faster_whisper_exe, project_root)}")
            local_model_folder = preflight_faster_whisper_model(project_root, effective_transcribe_model)
            if local_model_folder is not None:
                print(f"Preflight faster-whisper model: {_display_project_path(local_model_folder, project_root)}")
            else:
                print("Preflight faster-whisper model: built-in/downloadable model")

            effective_prompt = _effective_initial_prompt(args.initial_prompt, args.no_asmr_prompt)
            config = FasterWhisperConfig(
                model=effective_transcribe_model,
                language=args.src_lang,
                compute_type=args.transcribe_compute_type,
                model_dir=find_faster_whisper_model_dir(project_root),
                batched=args.fast_transcribe,
                batch_size=16,
                unmerged=args.fast_transcribe,
                initial_prompt=effective_prompt,
                reprompt=effective_prompt is not None,
            )

        progress = _ProgressDashboard(tasks, args.parallel, args.transcribe_parallel, started)
        _audio_to_idx = {task.audio_path: i for i, task in enumerate(tasks)}

        def _xxl_transcriber(*, audio_path, transcribe_srt_path, project_root, config):
            idx = _audio_to_idx.get(Path(audio_path))
            def _on_progress(ratio: float, cur: float, tot: float) -> None:
                if idx is not None:
                    progress.update_transcription_progress(idx, ratio, cur, tot)
            result = run_faster_whisper_transcription(
                audio_path=audio_path,
                transcribe_srt_path=transcribe_srt_path,
                project_root=project_root,
                config=config,
                transcription_progress_callback=_on_progress,
            )
            if idx is not None and not args.transcribe_only:
                progress.mark_translating(idx)
            return result

        progress.render()

        def translator(source_srt_path: Path, target_srt_path: Path) -> Path:
            return translate_srt_to_simplified(
                source_srt_path,
                target_srt_path,
                base_url=args.translate_api,
                model=args.translate_model,
                max_blocks=args.translate_chunk_size,
            )

        results = run_tasks(
            tasks,
            project_root=project_root,
            faster_whisper_config=config,
            translator=None if args.transcribe_only else translator,
            transcriber=transcriber or _xxl_transcriber,
            limit=args.limit,
            parallel=args.parallel,
            transcribe_parallel=args.transcribe_parallel,
            progress_callback=progress.update,
        )
        progress.stop()
        print_results_summary(results)
        return 0
    except Exception as exc:
        print(f"ERROR: {_safe_for_stdout(str(exc))}", file=sys.stderr)
        return 1


def print_results_summary(results: list[TaskResult]) -> None:
    success = sum(1 for result in results if result.status == "success")
    skipped = sum(1 for result in results if result.status == "skipped")
    failed = sum(1 for result in results if result.status == "failed")
    copied = sum(1 for result in results if result.status == "copied")
    print(f"Results: {success} success | {skipped} skipped | {failed} failed")
    if copied:
        print(f"Copied existing subtitles: {copied}")


def print_progress_snapshot(
    tasks: list[AudioTask],
    parallel: int,
    elapsed_seconds: float,
    transcribe_parallel: int = DEFAULT_TRANSCRIBE_PARALLEL,
    *,
    clear: bool = False,
    max_rows: int | None = None,
    task_started_at: list[float | None] | None = None,
    task_ended_at: list[float | None] | None = None,
    transcription_progress: dict[int, tuple[float, float, float]] | None = None,
) -> None:
    if clear and _stdout_is_tty():
        print("\033[2J\033[H", end="")
    total = len(tasks)
    queued = sum(1 for task in tasks if task.status == "queued")
    skipped = sum(1 for task in tasks if task.status == "skipped")
    failed = sum(1 for task in tasks if task.status == "failed")
    done = sum(1 for task in tasks if task.status in {"success", "copied", "transcribed", "skipped"})
    running = sum(1 for task in tasks if task.status == "running")
    elapsed = _format_elapsed(elapsed_seconds)
    print(
        f"Audio: {done}/{total} done | {running} running | {skipped} skipped | "
        f"{failed} failed | {queued} queued | parallel={parallel} | "
        f"transcribe_parallel={transcribe_parallel} | elapsed {elapsed}"
    )
    print()
    print(f"{'#':<4}{'Status':<10}{'Stage':<24}{'Time':>8}  Audio")
    visible = _visible_dashboard_tasks(tasks, max_rows=max_rows or parallel)
    for task_index, row_number, task in visible:
        audio_display = _safe_for_stdout(_truncate_text(task.relative_audio_path.as_posix(), 118))
        row_elapsed = _row_elapsed_text(task_index, task_started_at, task_ended_at)
        tp = (transcription_progress or {}).get(task_index - 1)
        print(f"{row_number:<4}{task.status:<10}{_stage_label(task, tp):<24}{row_elapsed:>8}  {audio_display}")
    sys.stdout.flush()


class _ProgressDashboard:
    _RENDER_INTERVAL = 0.25
    _HEARTBEAT_INTERVAL = 5.0
    _TERMINAL_STATUSES = {"success", "copied", "transcribed", "skipped", "failed"}

    def __init__(self, tasks: list[AudioTask], parallel: int, transcribe_parallel: int, started: float) -> None:
        self._tasks = tasks
        self._parallel = parallel
        self._transcribe_parallel = transcribe_parallel
        self._started = started
        self._statuses = [task.status for task in tasks]
        self._started_at: list[float | None] = [None] * len(tasks)
        self._ended_at: list[float | None] = [None] * len(tasks)
        self._transcription_progress: dict[int, tuple[float, float, float]] = {}
        self._lock = threading.Lock()
        self._last_render: float = 0.0
        self._stopped = threading.Event()
        self._heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat.start()

    def stop(self) -> None:
        self._stopped.set()

    def update_transcription_progress(self, index: int, ratio: float, current_s: float, total_s: float) -> None:
        with self._lock:
            self._transcription_progress[index] = (ratio, current_s, total_s)
            if monotonic() - self._last_render >= self._RENDER_INTERVAL:
                self._render_locked()

    def mark_translating(self, index: int) -> None:
        with self._lock:
            self._transcription_progress[index] = _TRANSLATING
            self._render_locked()

    def _heartbeat_loop(self) -> None:
        while not self._stopped.wait(self._HEARTBEAT_INTERVAL):
            with self._lock:
                self._render_locked()

    def update(self, index: int, task: AudioTask, status: str, result: TaskResult | None) -> None:
        del task, result
        with self._lock:
            now = monotonic()
            if 0 <= index < len(self._statuses):
                self._statuses[index] = status
                if status == "running" and self._started_at[index] is None:
                    self._started_at[index] = now
                if status in self._TERMINAL_STATUSES:
                    self._ended_at[index] = now
                    self._transcription_progress.pop(index, None)
            if status in self._TERMINAL_STATUSES or now - self._last_render >= self._RENDER_INTERVAL:
                self._render_locked()

    def render(self) -> None:
        with self._lock:
            self._render_locked()

    def _render_locked(self) -> None:
        self._last_render = monotonic()
        display_tasks = [replace(item, status=self._statuses[item_index]) for item_index, item in enumerate(self._tasks)]
        print_progress_snapshot(
            display_tasks,
            parallel=self._parallel,
            elapsed_seconds=monotonic() - self._started,
            transcribe_parallel=self._transcribe_parallel,
            clear=False,
            task_started_at=self._started_at,
            task_ended_at=self._ended_at,
            transcription_progress=dict(self._transcription_progress),
        )


def _visible_dashboard_tasks(tasks: list[AudioTask], *, max_rows: int) -> list[tuple[int, int, AudioTask]]:
    display_items: list[tuple[int, int, AudioTask]] = []
    row_number = 0
    for task_index, task in enumerate(tasks, start=1):
        if task.status == "skipped" and task.action == "skip":
            continue
        row_number += 1
        display_items.append((task_index, row_number, task))

    active_statuses = {"queued", "running"}
    first_active_position = next(
        (
            position
            for position, (_task_index, _row_number, task) in enumerate(display_items)
            if task.status in active_statuses or (task.status == "skipped" and task.action != "skip")
        ),
        None,
    )
    if first_active_position is None:
        window_start = max(0, len(display_items) - max_rows)
    else:
        window_start = first_active_position
    return display_items[window_start : window_start + max_rows]


def _stage_label(task: AudioTask, transcription_progress: tuple[float, float, float] | None = None) -> str:
    if task.status == "running" and task.action == "process":
        if transcription_progress == _TRANSLATING:
            return "translate"
        if transcription_progress is not None:
            ratio, cur, tot = transcription_progress
            return f"transcribe {int(ratio * 100)}% {_format_elapsed(cur)}/{_format_elapsed(tot)}"
        return "transcribe/translate"
    if task.action == "copy_existing_subtitle":
        return "copy-subtitle"
    if task.action == "copy_duplicate_audio_subtitle":
        return "copy-duplicate"
    if task.action == "skip":
        return "skip"
    return task.action


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _stdout_is_tty() -> bool:
    isatty = getattr(sys.stdout, "isatty", None)
    return bool(isatty and isatty())


def _row_elapsed_text(
    index: int,
    task_started_at: list[float | None] | None,
    task_ended_at: list[float | None] | None,
) -> str:
    if task_started_at is None:
        return "--:--"
    started = task_started_at[index - 1] if index - 1 < len(task_started_at) else None
    if started is None:
        return "--:--"
    ended = task_ended_at[index - 1] if task_ended_at is not None and index - 1 < len(task_ended_at) else None
    return _format_elapsed((ended or monotonic()) - started)


def _read_command_template(param_path: Path) -> str:
    try:
        return param_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return f"<missing: {param_path.as_posix()}>"


def _safe_for_stdout(text: str) -> str:
    encoding = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
    if encoding.startswith("utf") or "utf-" in encoding:
        return text
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def _format_elapsed(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _display_project_path(path: Path, project_root: Path) -> str:
    try:
        display = path.relative_to(project_root).as_posix()
    except ValueError:
        display = path.as_posix()
    return _safe_for_stdout(display)


def _effective_transcribe_model(backend: str, model: str | None) -> str:
    if model:
        return model
    if backend == "kotoba":
        return DEFAULT_KOTOBA_MODEL
    return DEFAULT_TRANSCRIBE_MODEL


def _transcription_mode_label(backend: str) -> str:
    if backend == "kotoba":
        return "kotoba python faster-whisper"
    return "local faster-whisper"


def _effective_initial_prompt(custom: str | None, no_asmr_prompt: bool) -> str | None:
    if custom is not None:
        return custom
    if no_asmr_prompt:
        return None
    return DEFAULT_ASMR_INITIAL_PROMPT
