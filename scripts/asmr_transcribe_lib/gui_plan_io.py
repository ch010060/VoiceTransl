"""Versioned JSON transport for a frozen plan; never rediscover or rematch it."""
from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from uuid import UUID, uuid5

from .gui_plan import (
    APP_NAMESPACE, DuplicateGroup, ExecutionPlan, PlanIssue, PlanRejected,
    PlannedTrack, PlannedWork, COMMON_SUBTITLE_EXTENSIONS, _hash, _namespace_issues,
    is_existing_subtitle_skip, path_key,
)
from .gui_results import atomic_json, read_json
from .models import AudioTask
from .planner import AUDIO_EXTENSIONS


class PlanTransportError(ValueError):
    pass


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _record(cls, value):
    if not isinstance(value, dict) or set(value) != {field.name for field in fields(cls)}:
        raise PlanTransportError("invalid frozen plan record")
    return dict(value)


def _audio_membership(plan):
    """Validate selected directory membership without invoking the matcher."""
    paths = set()
    for work in plan.works:
        if work.work_id not in plan.selected_work_ids:
            continue
        directory = plan.root / work.relative_path
        candidates = directory.iterdir() if work.kind == "synthetic_root" else directory.rglob("*")
        paths.update(path_key(path) for path in candidates
                     if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)
    return sorted(paths)


def _plan_hash(plan):
    return _hash({
        "settings": plan.settings, "selected": plan.selected_work_ids,
        "excluded": plan.excluded_track_ids,
        "tracks": [(t.track_id, t.task.action, t.dependencies, t.verification) for t in plan.tracks],
        **({"existing_subtitles": [(t.track_id, path_key(t.task.existing_subtitle_path))
                                   for t in plan.tracks if is_existing_subtitle_skip(t.task)]}
           if plan.settings.get("gui_existing_subtitles") == "skip_common" else {}),
        "groups": [(g.group_id, g.members, g.representative_track_id) for g in plan.duplicate_groups],
        "issues": [(i.code, i.track_ids, [str(p) for p in i.paths]) for i in plan.issues],
        "inventory": [(str(p), stamp) for p, stamp in plan.inventory],
    })


def _validate(plan):
    """Check transport identity/graph/path consistency without changing policy."""
    if not plan.root.is_absolute() or plan.mode not in {"full", "transcribe_only"}:
        raise PlanTransportError("invalid root or mode")
    expected_root = str(uuid5(APP_NAMESPACE, path_key(plan.root)))
    if plan.root_id != expected_root or plan.settings_hash != _hash(plan.settings):
        raise PlanTransportError("frozen identity or configuration mismatch")
    if plan.settings.get("path") != str(plan.root) or plan.settings.get("transcribe_only") != (plan.mode == "transcribe_only"):
        raise PlanTransportError("frozen configuration does not match plan")
    subtitle_policy = plan.settings.get("gui_existing_subtitles")
    if subtitle_policy not in {None, "skip_common"}:
        raise PlanTransportError("unsupported frozen subtitle policy")
    if plan.plan_hash != _plan_hash(plan) or plan.plan_id != str(uuid5(UUID(plan.root_id), "plan/" + plan.plan_hash)):
        raise PlanTransportError("frozen plan hash mismatch")
    work_ids = {work.work_id for work in plan.works}
    tracks = plan.track_index
    if len(work_ids) != len(plan.works) or len(tracks) != len(plan.tracks):
        raise PlanTransportError("duplicate frozen identity")
    if len(set(plan.selected_work_ids)) != len(plan.selected_work_ids) or set(plan.selected_work_ids) - work_ids or set(plan.excluded_track_ids) - tracks.keys():
        raise PlanTransportError("invalid selection")
    groups = {}
    for work in plan.works:
        if work.kind not in {"directory", "synthetic_root"}:
            raise PlanTransportError("invalid Work kind")
        relative = Path(work.relative_path)
        if relative.is_absolute() or ".." in relative.parts or (work.kind == "directory" and len(relative.parts) != 1) or (work.kind == "synthetic_root" and work.relative_path):
            raise PlanTransportError("invalid Work path")
        identity = "synthetic-root" if work.kind == "synthetic_root" else "work/" + work.relative_path
        if work.work_id != str(uuid5(UUID(plan.root_id), identity)) or work.track_ids != tuple(t.track_id for t in plan.tracks if t.work_id == work.work_id):
            raise PlanTransportError("invalid Work identity or membership")
    for group in plan.duplicate_groups:
        if not group.members or group.representative_track_id not in group.members or group.work_id not in work_ids:
            raise PlanTransportError("invalid dependency group")
        if group.group_id != str(uuid5(UUID(group.work_id), "group/" + "|".join(group.members))):
            raise PlanTransportError("invalid group identity")
        for member in group.members:
            if member in groups or member not in tracks or tracks[member].work_id != group.work_id:
                raise PlanTransportError("invalid group membership")
            groups[member] = group
    for ordinal, track in enumerate(plan.tracks):
        task = track.task
        relative = task.audio_path.relative_to(plan.root)
        if not task.audio_path.is_absolute() or not task.audio_path.resolve().is_relative_to(plan.root.resolve()) or task.audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise PlanTransportError("audio path escapes frozen root")
        if track.ordinal != ordinal or track.track_id != str(uuid5(UUID(plan.root_id), "track/" + relative.as_posix())) or task.relative_audio_path != relative or task.root_relative_audio_path != relative:
            raise PlanTransportError("invalid Track identity")
        work = next((w for w in plan.works if w.work_id == track.work_id), None)
        if work is None or task.work_dir != plan.root / work.relative_path or not task.audio_path.resolve().is_relative_to(task.work_dir.resolve()):
            raise PlanTransportError("Track outside Work")
        expected = (task.audio_path.with_suffix(".srt"),) + tuple(task.audio_path.with_name(task.audio_path.stem + suffix) for suffix in ("-transcribe.srt", "-translate.srt", "-opencc.srt"))
        if expected != (task.final_srt_path, task.transcribe_srt_path, task.translate_srt_path, task.opencc_srt_path):
            raise PlanTransportError("output namespace changed")
        if any(not path.resolve().is_relative_to(plan.root.resolve()) for path in expected):
            raise PlanTransportError("output namespace escapes root")
        group = groups.get(track.track_id)
        if group is None or track.representative_track_id != group.representative_track_id or task.action not in {"process", "skip", "copy_existing_subtitle", "copy_duplicate_audio_subtitle"}:
            raise PlanTransportError("invalid frozen action or representative")
        if len(track.dependencies) > 1 or any(dep not in tracks or dep == track.track_id or tracks[dep].work_id != track.work_id for dep in track.dependencies):
            raise PlanTransportError("invalid Track dependency")
        if task.action == "copy_duplicate_audio_subtitle" and track.dependencies != (group.representative_track_id,):
            raise PlanTransportError("duplicate dependency changed")
        if task.existing_subtitle_path and not task.existing_subtitle_path.resolve().is_relative_to(task.work_dir.resolve()):
            raise PlanTransportError("subtitle dependency escapes Work")
        if is_existing_subtitle_skip(task):
            source = task.existing_subtitle_path
            if (subtitle_policy != "skip_common" or track.dependencies or task.status != "skipped"
                    or track.verification != "unverified_legacy" or not source or not source.is_file()
                    or source.suffix.lower() not in COMMON_SUBTITLE_EXTENSIONS):
                raise PlanTransportError("invalid frozen existing subtitle skip")
            members = [tracks[member].task.audio_path for member in group.members]
            if not any(source.parent == audio.parent and source.stem.casefold() == audio.stem.casefold()
                       for audio in members):
                raise PlanTransportError("existing subtitle outside duplicate group")
        if track.dependencies:
            owner = tracks[track.dependencies[0]].task
            source = owner.transcribe_srt_path if plan.mode == "transcribe_only" else owner.final_srt_path
            if (subtitle_policy == "skip_common" and is_existing_subtitle_skip(owner)
                    and owner.existing_subtitle_path and owner.existing_subtitle_path.suffix.lower() == ".srt"):
                source = owner.final_srt_path
            if task.existing_subtitle_path != source:
                raise PlanTransportError("subtitle dependency changed")
        seen = {track.track_id}
        pending = list(track.dependencies)
        while pending:
            dependency = pending.pop()
            if dependency in seen:
                raise PlanTransportError("cyclic dependency")
            seen.add(dependency)
            pending.extend(tracks[dependency].dependencies)
    selected = [t for t in plan.tracks if t.work_id in plan.selected_work_ids]
    if plan.source_inventory_fingerprint != _hash([(t.track_id, t.source_stamp) for t in selected]):
        raise PlanTransportError("source fingerprint mismatch")
    for path, stamp in plan.inventory:
        if not path.resolve().is_relative_to(plan.root.resolve()) or not stamp or stamp[0] != path_key(path):
            raise PlanTransportError("invalid inventory path")
    if _namespace_issues(plan.planned_tracks, plan.duplicate_groups, plan.mode) and not plan.issues:
        raise PlanTransportError("unsafe frozen output namespace")


def encode_plan(plan: ExecutionPlan) -> dict:
    _validate(plan)
    expected = sorted(path_key(t.task.audio_path) for t in plan.tracks if t.work_id in plan.selected_work_ids)
    if _audio_membership(plan) != expected:
        raise PlanRejected((PlanIssue("PLAN_STALE", (), detail="audio membership changed"),))
    payload = {"schema_version": "1.0", "plan": _json_value(asdict(plan)), "audio_membership": expected}
    return {**payload, "transport_hash": _hash(payload)}


def decode_plan(document: dict, expected_digest: str | None = None) -> ExecutionPlan:
    try:
        if not isinstance(document, dict) or set(document) != {"schema_version", "plan", "audio_membership", "transport_hash"} or document["schema_version"] != "1.0":
            raise PlanTransportError("unsupported frozen plan schema")
        digest = _hash({key: value for key, value in document.items() if key != "transport_hash"})
        if digest != document["transport_hash"] or (expected_digest is not None and digest != expected_digest):
            raise PlanTransportError("frozen transport digest mismatch")
        record = _record(ExecutionPlan, document["plan"])
        record["root"] = Path(record["root"])
        record["works"] = tuple(PlannedWork(**{**_record(PlannedWork, w), "track_ids": tuple(w["track_ids"])}) for w in record["works"])
        parsed = []
        for track in record["tracks"]:
            entry = _record(PlannedTrack, track)
            task = _record(AudioTask, entry["task"])
            for key in ("work_dir", "audio_path", "root_relative_audio_path", "relative_audio_path", "final_srt_path", "transcribe_srt_path", "translate_srt_path", "opencc_srt_path", "existing_subtitle_path"):
                task[key] = Path(task[key]) if task[key] is not None else None
            entry.update(task=AudioTask(**task), source_stamp=tuple(entry["source_stamp"]), dependencies=tuple(entry["dependencies"]))
            parsed.append(PlannedTrack(**entry))
        record["tracks"] = tuple(parsed)
        record["duplicate_groups"] = tuple(DuplicateGroup(**{**_record(DuplicateGroup, g), "members": tuple(g["members"])}) for g in record["duplicate_groups"])
        record["issues"] = tuple(PlanIssue(**{**_record(PlanIssue, i), "track_ids": tuple(i["track_ids"]), "paths": tuple(map(Path, i["paths"]))}) for i in record["issues"])
        record["inventory"] = tuple((Path(path), tuple(stamp)) for path, stamp in record["inventory"])
        for key in ("selected_work_ids", "excluded_track_ids"):
            record[key] = tuple(record[key])
        plan = ExecutionPlan(**record)
        _validate(plan)
        expected = sorted(path_key(t.task.audio_path) for t in plan.tracks if t.work_id in plan.selected_work_ids)
        if document["audio_membership"] != expected or _audio_membership(plan) != expected:
            raise PlanTransportError("frozen audio membership changed")
        return plan
    except (TypeError, KeyError, AttributeError, OSError, ValueError) as exc:
        if isinstance(exc, PlanTransportError):
            raise
        raise PlanTransportError("invalid frozen plan") from exc


def save_plan(plan: ExecutionPlan, path: Path | str) -> str:
    path = Path(path)
    if path.resolve().is_relative_to(plan.root.resolve()):
        raise PlanTransportError("plan storage must be outside audio library")
    document = encode_plan(plan)
    if path.exists():
        raise PlanTransportError("refusing to replace frozen plan")
    atomic_json(path, document)
    return document["transport_hash"]


def load_plan(path: Path | str, expected_digest: str | None = None) -> ExecutionPlan:
    return decode_plan(read_json(path), expected_digest)
