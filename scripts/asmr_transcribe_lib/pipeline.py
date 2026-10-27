from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import shutil
import threading

from .local_faster_whisper import FasterWhisperConfig, run_faster_whisper_transcription
from .models import AudioTask
from .opencc_utils import convert_srt_file_opencc
from .srt_utils import parse_srt_file, validate_same_timing
from .transcript_corrections import apply_default_traditional_chinese_corrections, apply_default_transcript_corrections


@dataclass(frozen=True)
class TaskResult:
    status: str
    reason: str
    audio_path: Path
    output_path: Path | None = None


ProgressCallback = Callable[[int, AudioTask, str, TaskResult | None], None]
ProgressTickCallback = Callable[[], None]


def process_audio_task(
    task: AudioTask,
    *,
    project_root: Path | str,
    faster_whisper_config: FasterWhisperConfig,
    transcriber: Callable[..., Path] = run_faster_whisper_transcription,
    translator: Callable[[Path, Path], Path] | None = None,
    opencc_converter: Callable[..., Path] = convert_srt_file_opencc,
) -> TaskResult:
    if task.final_srt_path.exists():
        return TaskResult(
            status="skipped",
            reason="existing_srt",
            audio_path=task.audio_path,
            output_path=task.final_srt_path,
        )

    if task.action in {"copy_existing_subtitle", "copy_duplicate_audio_subtitle"}:
        if task.existing_subtitle_path is None:
            raise FileNotFoundError(f"existing subtitle path missing for {task.audio_path}")
        if not task.existing_subtitle_path.exists():
            raise FileNotFoundError(f"subtitle source does not exist yet for {task.audio_path}: {task.existing_subtitle_path}")
        if task.final_srt_path.exists():
            return TaskResult(
                status="skipped",
                reason="existing_srt",
                audio_path=task.audio_path,
                output_path=task.final_srt_path,
            )
        task.final_srt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(task.existing_subtitle_path, task.final_srt_path)
        return TaskResult(
            status="copied",
            reason=task.reason or "existing_alt_subtitle",
            audio_path=task.audio_path,
            output_path=task.final_srt_path,
        )

    if task.status == "skipped":
        return TaskResult(
            status="skipped",
            reason=task.reason or "skipped",
            audio_path=task.audio_path,
            output_path=None,
        )

    if task.transcribe_srt_path.exists() and _is_valid_srt(task.transcribe_srt_path):
        for _intermediate in [task.translate_srt_path, task.opencc_srt_path]:
            if _intermediate.exists():
                _intermediate.unlink()
        transcribed_path = task.transcribe_srt_path
    else:
        _remove_intermediates(task)
        transcribed_path = transcriber(
            audio_path=task.audio_path,
            transcribe_srt_path=task.transcribe_srt_path,
            project_root=Path(project_root),
            config=faster_whisper_config,
        )
    transcribed_path = apply_default_transcript_corrections(transcribed_path, project_root=project_root)
    transcribed_blocks = parse_srt_file(transcribed_path)

    if translator is None:
        return TaskResult(
            status="transcribed",
            reason="transcribe_complete",
            audio_path=task.audio_path,
            output_path=Path(transcribed_path),
        )

    translated_path = translator(Path(transcribed_path), task.translate_srt_path)
    translated_blocks = parse_srt_file(translated_path)
    validate_same_timing(transcribed_blocks, translated_blocks)

    opencc_path = opencc_converter(Path(translated_path), task.opencc_srt_path)
    opencc_path = apply_default_traditional_chinese_corrections(opencc_path, project_root=project_root)
    opencc_blocks = parse_srt_file(opencc_path)
    validate_same_timing(translated_blocks, opencc_blocks)

    if task.final_srt_path.exists():
        return TaskResult(
            status="skipped",
            reason="existing_srt",
            audio_path=task.audio_path,
            output_path=task.final_srt_path,
        )
    _finalize_opencc_srt(Path(opencc_path), task.final_srt_path)
    return TaskResult(
        status="success",
        reason="pipeline_complete",
        audio_path=task.audio_path,
        output_path=task.final_srt_path,
    )


def run_tasks(
    tasks: list[AudioTask],
    *,
    project_root: Path | str,
    faster_whisper_config: FasterWhisperConfig,
    translator: Callable[[Path, Path], Path],
    transcriber: Callable[..., Path] = run_faster_whisper_transcription,
    opencc_converter: Callable[..., Path] = convert_srt_file_opencc,
    limit: int | None = None,
    parallel: int = 1,
    transcribe_parallel: int = 1,
    progress_callback: ProgressCallback | None = None,
    progress_tick_callback: ProgressTickCallback | None = None,
) -> list[TaskResult]:
    if parallel < 1:
        raise ValueError("parallel must be >= 1")
    if transcribe_parallel < 1:
        raise ValueError("transcribe_parallel must be >= 1")
    transcribe_semaphore = threading.Semaphore(transcribe_parallel)
    if parallel == 1:
        return _run_tasks_sequential(
            tasks,
            project_root=project_root,
            faster_whisper_config=faster_whisper_config,
            translator=translator,
            transcriber=transcriber,
            transcribe_semaphore=transcribe_semaphore,
            opencc_converter=opencc_converter,
            limit=limit,
            progress_callback=progress_callback,
        )

    indexed_results: list[tuple[int, TaskResult]] = []
    futures: list[tuple[int, Future[TaskResult]]] = []
    deferred_copies: list[tuple[int, AudioTask]] = []
    processed_audio_count = 0

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        for index, task in enumerate(tasks):
            consumes_limit = task.status == "queued" and task.action == "process"
            if limit is not None and consumes_limit and processed_audio_count >= limit:
                break

            if consumes_limit:
                future = executor.submit(
                    _process_with_transcribe_gate,
                    index,
                    task,
                    progress_callback,
                    project_root=project_root,
                    faster_whisper_config=faster_whisper_config,
                    transcriber=transcriber,
                    transcribe_semaphore=transcribe_semaphore,
                    translator=translator,
                    opencc_converter=opencc_converter,
                )
                futures.append(
                    (
                        index,
                        future,
                    )
                )
                processed_audio_count += 1
            elif _should_defer_duplicate_copy(task):
                deferred_copies.append((index, task))
            else:
                indexed_results.append(
                    (
                        index,
                        process_audio_task(
                            task,
                            project_root=project_root,
                            faster_whisper_config=faster_whisper_config,
                            transcriber=transcriber,
                            translator=translator,
                            opencc_converter=opencc_converter,
                        ),
                    )
                )

        future_indexes = {future: index for index, future in futures}
        while future_indexes:
            done_futures, _pending = wait(future_indexes, timeout=1.0, return_when=FIRST_COMPLETED)
            if not done_futures:
                if progress_tick_callback is not None:
                    progress_tick_callback()
                continue
            for future in done_futures:
                index = future_indexes.pop(future)
                result = future.result()
                indexed_results.append((index, result))
                if progress_callback is not None:
                    progress_callback(index, tasks[index], result.status, result)

        for index, task in deferred_copies:
            indexed_results.append(
                (
                    index,
                    process_audio_task(
                        task,
                        project_root=project_root,
                        faster_whisper_config=faster_whisper_config,
                        transcriber=transcriber,
                        translator=translator,
                        opencc_converter=opencc_converter,
                    ),
                )
            )

    return [result for _index, result in sorted(indexed_results, key=lambda item: item[0])]


def _run_tasks_sequential(
    tasks: list[AudioTask],
    *,
    project_root: Path | str,
    faster_whisper_config: FasterWhisperConfig,
    translator: Callable[[Path, Path], Path],
    transcriber: Callable[..., Path],
    transcribe_semaphore: threading.Semaphore,
    opencc_converter: Callable[..., Path],
    limit: int | None,
    progress_callback: ProgressCallback | None,
) -> list[TaskResult]:
    results: list[TaskResult] = []
    deferred_copies: list[tuple[int, AudioTask]] = []
    processed_audio_count = 0

    for task_index, task in enumerate(tasks):
        consumes_limit = task.status == "queued" and task.action == "process"
        if limit is not None and consumes_limit and processed_audio_count >= limit:
            break

        if _should_defer_duplicate_copy(task):
            deferred_copies.append((task_index, task))
            continue

        result = _process_with_transcribe_gate(
            task_index,
            task,
            progress_callback,
            project_root=project_root,
            faster_whisper_config=faster_whisper_config,
            transcriber=transcriber,
            transcribe_semaphore=transcribe_semaphore,
            translator=translator,
            opencc_converter=opencc_converter,
        )
        results.append(result)
        if progress_callback is not None:
            progress_callback(task_index, task, result.status, result)

        if consumes_limit:
            processed_audio_count += 1

    for task_index, task in deferred_copies:
        result = _process_with_transcribe_gate(
            task_index,
            task,
            progress_callback,
            project_root=project_root,
            faster_whisper_config=faster_whisper_config,
            transcriber=transcriber,
            transcribe_semaphore=transcribe_semaphore,
            translator=translator,
            opencc_converter=opencc_converter,
        )
        results.append(result)
        if progress_callback is not None:
            progress_callback(task_index, task, result.status, result)

    return results


def _process_with_transcribe_gate(
    index: int,
    task: AudioTask,
    progress_callback: ProgressCallback | None,
    *,
    project_root: Path | str,
    faster_whisper_config: FasterWhisperConfig,
    transcriber: Callable[..., Path],
    transcribe_semaphore: threading.Semaphore,
    translator: Callable[[Path, Path], Path],
    opencc_converter: Callable[..., Path],
) -> TaskResult:
    def gated_task_transcriber(**kwargs) -> Path:
        with transcribe_semaphore:
            if progress_callback is not None:
                progress_callback(index, task, "running", None)
            return transcriber(**kwargs)

    return process_audio_task(
        task,
        project_root=project_root,
        faster_whisper_config=faster_whisper_config,
        transcriber=gated_task_transcriber,
        translator=translator,
        opencc_converter=opencc_converter,
    )


def _should_defer_duplicate_copy(task: AudioTask) -> bool:
    return (
        task.action == "copy_duplicate_audio_subtitle"
        and task.existing_subtitle_path is not None
        and not task.existing_subtitle_path.exists()
    )


def _limit_transcriber_concurrency(
    transcriber: Callable[..., Path],
    transcribe_parallel: int,
) -> Callable[..., Path]:
    if transcribe_parallel == 1:
        semaphore = threading.Semaphore(1)

        def locked_transcriber(**kwargs) -> Path:
            with semaphore:
                return transcriber(**kwargs)

        return locked_transcriber

    semaphore = threading.Semaphore(transcribe_parallel)

    def limited_transcriber(**kwargs) -> Path:
        with semaphore:
            return transcriber(**kwargs)

    return limited_transcriber


def _is_valid_srt(path: Path) -> bool:
    try:
        return len(parse_srt_file(path)) > 0
    except Exception:
        return False


def _remove_intermediates(task: AudioTask) -> None:
    for path in [task.transcribe_srt_path, task.translate_srt_path, task.opencc_srt_path]:
        if path.exists():
            path.unlink()


def _finalize_opencc_srt(opencc_srt_path: Path, final_srt_path: Path) -> None:
    final_srt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_final = final_srt_path.with_name(f"{final_srt_path.name}.tmp")
    if tmp_final.exists():
        tmp_final.unlink()
    try:
        shutil.copyfile(opencc_srt_path, tmp_final)
        tmp_final.replace(final_srt_path)
    except Exception:
        if tmp_final.exists():
            tmp_final.unlink()
        raise
