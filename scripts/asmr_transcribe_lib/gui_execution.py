"""Phase 1A opt-in safe-plan entry point; scheduling remains in pipeline.run_tasks."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import inspect
import hashlib
import os
from pathlib import Path
import shutil
import threading

from .gui_plan import (ExecutionPlan, Provenance, file_stamp, is_existing_subtitle_skip,
                       make_provenance, path_key, validate_srt)
from .local_faster_whisper import FasterWhisperConfig, find_faster_whisper_model_dir, run_faster_whisper_transcription
from .pipeline import TaskResult, process_audio_task, run_tasks
from .runtime import CancellationToken, ObserverError, PipelineCancelled, PipelineEvent, TaskRuntime
from .srt_normalization import normalize_srt_bytes, TimestampOrderError, WARNING_CODE


@dataclass(frozen=True)
class TrackOutcome(TaskResult):
    track_id: str = ""
    warnings: tuple[str, ...] = ()
    provenance: tuple[Provenance, ...] = ()


def _publish_new(source: Path, target: Path) -> None:
    """Same-volume atomic publication without replacing an existing target.

    Exclusive temp creation also refuses stale/foreign temps. Cross-instance
    ownership is still a separate Phase 1B gate, not claimed by this helper.
    """
    validate_srt(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    created = False
    try:
        with tmp.open("xb") as output:
            created = True
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
        os.link(tmp, target)  # Unlike replace(), fails if a final appeared late.
    finally:
        if created:
            tmp.unlink(missing_ok=True)


def _config(plan: ExecutionPlan, project_root: Path):
    s = plan.settings
    if s["transcribe_backend"] == "kotoba":
        from .kotoba_faster_whisper import KotobaFasterWhisperConfig
        return KotobaFasterWhisperConfig(
            model=s["transcribe_model"], language=s["src_lang"],
            device=s["kotoba_device"], compute_type=s["transcribe_compute_type"],
            chunk_length=s["kotoba_chunk_length"], condition_on_previous_text=False,
        )
    return FasterWhisperConfig(
        model=s["transcribe_model"],
        language=s["src_lang"], compute_type=s["transcribe_compute_type"],
        model_dir=find_faster_whisper_model_dir(project_root),
        batched=s["fast_transcribe"], batch_size=s["transcribe_batch_size"],
        sentence=s["fast_transcribe"], chunk_length=s["xxl_chunk_length"],
        initial_prompt=s["initial_prompt"], reprompt=s["initial_prompt"] is not None,
    )


def run_plan(plan: ExecutionPlan, *, project_root: Path | str,
             translator: Callable | None, transcriber: Callable = run_faster_whisper_transcription,
             opencc_converter: Callable, parallel: int | None = None,
             observer: Callable[[PipelineEvent], None] | None = None,
             cancel_token: CancellationToken | None = None,
             normalization_store: Callable | None = None) -> list[TrackOutcome]:
    """Execute a checked plan through the existing scheduler (no new scheduler).

    Ordinary failures are isolated only here. Legacy run_tasks callers retain
    their raise/CLI-exit semantics unless they explicitly supply a processor.
    """
    plan.assert_ready()
    settings = plan.settings
    if parallel is not None and parallel != settings["parallel"]:
        raise ValueError("execution parallel differs from frozen plan settings")
    if (plan.mode == "full") != (translator is not None):
        raise ValueError("translator does not match frozen plan mode")
    if settings["transcribe_backend"] != "xxl" and transcriber is run_faster_whisper_transcription:
        raise ValueError("selected backend requires an explicit compatible transcriber")
    token = cancel_token or CancellationToken()
    lock = threading.RLock()
    outcomes: dict[str, TrackOutcome] = {}
    planned = {track.task.audio_path: track for track in plan.planned_tracks}

    def emit(event: PipelineEvent) -> None:
        if observer is not None:
            with lock:
                try:
                    observer(event)
                except Exception as exc:
                    token.cancel()
                    raise ObserverError("observer failed") from exc

    def process(*, index, task, progress_callback, project_root, faster_whisper_config,
                transcriber, transcribe_semaphore, translator, opencc_converter):
        track = planned[task.audio_path]
        started = False
        receipts = []
        timestamp_warnings = []
        normalized_stamp = None
        stage = None
        expected_inventory = dict(plan.inventory)

        def unchanged(path):
            if file_stamp(path) != expected_inventory[path]:
                raise ValueError("planned output changed before execution")

        def stage_changed(name):
            nonlocal stage
            stage = name
            if name == "transcript_correction" and normalized_stamp is not None:
                if file_stamp(task.transcribe_srt_path) != normalized_stamp:
                    raise ValueError("normalized ASR output changed before correction")
            emit(PipelineEvent("stage_changed", track.track_id, name, "running"))
            if name == "waiting_asr":
                for path in [task.transcribe_srt_path, task.translate_srt_path, task.opencc_srt_path]:
                    unchanged(path)

        def finish(status, reason, output=None, warnings=()):
            warnings = tuple(dict.fromkeys((*warnings, *timestamp_warnings)))
            result = TrackOutcome(status, reason, task.audio_path, output, track.track_id,
                                  warnings, tuple(receipts))
            with lock:
                outcomes[track.track_id] = result
            event = {"failed": "track_failed", "cancelled": "track_cancelled",
                     "not_started": "track_not_started"}.get(status, "track_finished")
            error_reason = reason if reason.startswith("SRT_TIMESTAMP_ORDER_INVALID:") else "track_exception"
            emit(PipelineEvent(event, track.track_id, stage, status,
                               reason=reason if status != "failed" else error_reason, warnings=warnings))
            return result

        def validate(path):
            validate_srt(path)
            kind = {path_key(p): role for role, p in [
                ("transcription", task.transcribe_srt_path), ("translation", task.translate_srt_path),
                ("opencc", task.opencc_srt_path)]}.get(path_key(path))
            if kind:
                receipts.append(make_provenance(plan, track.track_id, path, kind))

        def check_normalized_outputs():
            if timestamp_warnings:
                token.checkpoint()
                if file_stamp(task.audio_path) != track.source_stamp:
                    raise ValueError("source changed after normalization")
                for proof in receipts:
                    if hashlib.sha256(Path(proof.output_key).read_bytes()).hexdigest() != proof.output_hash:
                        raise ValueError("validated stage output changed after normalization")

        def publish(source, target):
            check_normalized_outputs()
            _publish_new(source, target)

        def gated_transcriber(**kwargs):
            nonlocal normalized_stamp
            while not transcribe_semaphore.acquire(timeout=0.05):
                token.checkpoint()
            try:
                token.checkpoint()
                stage_changed("asr")
                token.checkpoint()
                if file_stamp(task.audio_path) != track.source_stamp:
                    raise ValueError("source changed while waiting for ASR")
                if task.transcribe_srt_path.exists():
                    raise ValueError("unexpected transcript appeared before ASR")
                if progress_callback:
                    progress_callback(index, task, "running", None)
                def report(ratio, current, total):
                    emit(PipelineEvent("asr_progress", track.track_id, "asr", "running", (ratio, current, total)))
                parameters = inspect.signature(transcriber).parameters
                if "transcription_progress_callback" in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
                    kwargs["transcription_progress_callback"] = report
                result = transcriber(**kwargs)
                if path_key(Path(result)) != path_key(task.transcribe_srt_path):
                    raise ValueError("ASR returned an unowned output path")
                token.checkpoint()
                if file_stamp(task.audio_path) != track.source_stamp:
                    raise ValueError("source changed during ASR")
                try:
                    validate_srt(Path(result))
                except ValueError:
                    stamp = file_stamp(Path(result))
                    raw = Path(result).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != stamp[-1]:
                        raise ValueError("ASR output changed while reading")
                    normalized, changes = normalize_srt_bytes(raw)
                    if not changes:
                        raise TimestampOrderError("strict_validation")
                    if normalization_store is None:
                        raise TimestampOrderError("audit_store_required")
                    normalization_store(track.track_id, raw, normalized, changes)
                    temporary = task.transcribe_srt_path.with_name(task.transcribe_srt_path.name + ".tmp")
                    created = False
                    try:
                        with temporary.open("xb") as output:
                            created = True
                            output.write(normalized)
                            output.flush()
                            os.fsync(output.fileno())
                        validate_srt(temporary)
                        token.checkpoint()
                        if (file_stamp(task.audio_path) != track.source_stamp or
                                file_stamp(Path(result)) != stamp):
                            raise ValueError("source or ASR output changed before normalization publication")
                        os.replace(temporary, result)
                        normalized_stamp = file_stamp(Path(result))
                        if normalized_stamp[-1] != hashlib.sha256(normalized).hexdigest():
                            raise ValueError("normalized ASR output changed during publication")
                    finally:
                        if created:
                            temporary.unlink(missing_ok=True)
                    timestamp_warnings.append(WARNING_CODE)
                emit(PipelineEvent("asr_completed", track.track_id, "asr", "running"))
                return result
            finally:
                transcribe_semaphore.release()

        try:
            if token.cancelled:
                return finish("not_started", "run_cancelled")
            for dependency in track.dependencies:
                with lock:
                    source_result = outcomes.get(dependency)
                if source_result is None or source_result.status not in {"success", "transcribed", "copied", "skipped"}:
                    return finish("not_started", "dependency_failed")
            if file_stamp(task.audio_path) != track.source_stamp:
                return finish("not_started", "source_changed")
            if task.action == "skip":
                output = task.existing_subtitle_path if is_existing_subtitle_skip(task) else task.final_srt_path
                unchanged(output)
                stage_changed("skip")
                warning = ("unverified_legacy",) if track.verification == "unverified_legacy" else ()
                return finish("skipped", task.reason if is_existing_subtitle_skip(task) else "existing_srt", output, warning)
            started = True
            emit(PipelineEvent("track_started", track.track_id, status="running"))
            for path in [task.final_srt_path, task.transcribe_srt_path, task.translate_srt_path, task.opencc_srt_path]:
                # A dependency may legitimately have just committed a shared output.
                if not track.dependencies:
                    unchanged(path)
            effective_task = task
            if task.action in {"copy_duplicate_audio_subtitle", "copy_existing_subtitle"}:
                target = task.transcribe_srt_path if plan.mode == "transcribe_only" else task.final_srt_path
                effective_task = replace(task, final_srt_path=target)
                token.checkpoint()
                if track.shared_output_owner:
                    # The dependency has already published the single physical
                    # output. This Track certifies a read, never copies/replaces.
                    if (source_result.output_path is None or
                            path_key(source_result.output_path) != path_key(target)):
                        raise ValueError("shared output owner returned a different output")
                    stage_changed("copy")
                    token.checkpoint()
                    validate_srt(target)
                    receipts.append(make_provenance(plan, track.track_id, target,
                                                   "transcription" if plan.mode == "transcribe_only" else "final"))
                    return finish("copied", "shared_representative_output", target)
                if task.existing_subtitle_path and path_key(task.existing_subtitle_path) == path_key(target):
                    stage_changed("copy")
                    token.checkpoint()
                    validate_srt(target)
                    return finish("copied", "shared_representative_output", target)
            elif task.transcribe_srt_path.exists():
                emit(PipelineEvent("asr_reused", track.track_id, reason="matching_provenance"))
            runtime = TaskRuntime(stage_changed, token.checkpoint, validate, publish)
            def checked_translator(source, target):
                token.checkpoint()
                output = translator(source, target)
                if path_key(Path(output)) != path_key(target):
                    raise ValueError("translator returned an unowned output path")
                return output

            def checked_opencc(source, target):
                token.checkpoint()
                output = opencc_converter(source, target)
                if path_key(Path(output)) != path_key(target):
                    raise ValueError("OpenCC returned an unowned output path")
                return output
            result = process_audio_task(
                effective_task, project_root=project_root, faster_whisper_config=faster_whisper_config,
                transcriber=gated_transcriber, translator=checked_translator if translator is not None else None,
                opencc_converter=checked_opencc, runtime=runtime,
            )
            if result.output_path is not None and result.status in {"success", "copied", "transcribed"}:
                check_normalized_outputs()
                validate_srt(result.output_path)
                receipts.append(make_provenance(plan, track.track_id, result.output_path,
                                               "transcription" if plan.mode == "transcribe_only" else "final"))
            if result.status == "skipped" and result.output_path is not None:
                unchanged(result.output_path)
            return finish(result.status, result.reason, result.output_path)
        except ObserverError:
            raise
        except PipelineCancelled:
            return finish("cancelled" if started else "not_started", "run_cancelled")
        except TimestampOrderError as exc:
            return finish("failed", str(exc))
        except Exception as exc:
            return finish("failed", f"{type(exc).__name__}: {exc}")

    for track in plan.planned_tracks:
        emit(PipelineEvent("track_queued", track.track_id, status="queued"))
    # Stable dependency order within the existing scheduler's deferred-copy
    # queue. Keep ordinals/IDs/result ordering from the frozen plan unchanged.
    ordered, visited = [], set()
    planned_ids = {t.track_id for t in plan.planned_tracks}
    def visit(track):
        if track.track_id in visited:
            return
        visited.add(track.track_id)
        for dependency in track.dependencies:
            if dependency not in planned_ids:
                raise ValueError("dependency is outside planned execution")
            visit(plan.track_index[dependency])
        ordered.append(track.task)
    for track in plan.planned_tracks:
        visit(track)
    results = run_tasks(
        ordered, project_root=project_root,
        faster_whisper_config=_config(plan, Path(project_root)), translator=translator,
        transcriber=transcriber, opencc_converter=opencc_converter,
        parallel=settings["parallel"], transcribe_parallel=settings["transcribe_parallel"],
        task_processor=process,
        defer_task=lambda task: bool(planned[task.audio_path].dependencies),
    )
    by_audio = {r.audio_path: r for r in results}
    return [by_audio[t.task.audio_path] for t in plan.planned_tracks]
