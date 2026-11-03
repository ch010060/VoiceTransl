"""Frozen, opt-in GUI-safe plans over the unmodified ASMR filename matcher.

No scheduler, model loading, audio equivalence gate, or persistent journal lives
here. Provenance records are supplied by a trusted completed-stage owner; merely
parsing an old file never creates one automatically.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from functools import cached_property
from types import MappingProxyType
from uuid import UUID, uuid5

from .models import AudioTask
from .planner import (
    discover_audio_tasks, _choose_representative_audio, _group_same_content_tasks,
)
from .srt_utils import parse_srt_file

APP_NAMESPACE = UUID("5f3ddd82-3f63-4fb0-9ec4-fc9d8d62a501")
COMMON_SUBTITLE_EXTENSIONS = (".srt", ".vtt", ".ass", ".ssa", ".lrc")


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def file_stamp(path: Path) -> tuple:
    if not path.exists():
        return (path_key(path), None)
    stat = path.stat()
    # Audio identity/freshness metadata only: NOT an equivalence gate for variants.
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() and path.suffix.lower() in {*COMMON_SUBTITLE_EXTENSIONS, ".tmp"} else None
    return (path_key(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, digest)


def _existing_common_subtitle(task: AudioTask) -> Path | None:
    candidates = [path for path in task.audio_path.parent.iterdir()
                  if path.stem.casefold() == task.audio_path.stem.casefold()
                  and path.suffix.lower() in COMMON_SUBTITLE_EXTENSIONS]
    return min(candidates, key=lambda path: (COMMON_SUBTITLE_EXTENSIONS.index(path.suffix.lower()),
                                           path.name.casefold(), path.name), default=None)


def is_existing_subtitle_skip(task: AudioTask) -> bool:
    return task.action == "skip" and task.reason == "existing_subtitle_skip"


def validate_srt(path: Path) -> None:
    blocks = parse_srt_file(path)
    if not blocks:
        raise ValueError("empty SRT")
    previous = -1
    for block in blocks:
        if not str(block.index).isdigit():
            raise ValueError("non-numeric SRT index")
        start, end = block.timestamp.split(" --> ")
        def milliseconds(value: str) -> int:
            h, m, tail = value.split(":")
            s, ms = tail.split(",")
            if int(m) >= 60 or int(s) >= 60:
                raise ValueError("SRT timestamp out of range")
            return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)
        left, right = milliseconds(start), milliseconds(end)
        if right < left or left < previous:
            raise ValueError("SRT timestamps out of order")
        previous = left


@dataclass(frozen=True)
class PlanIssue:
    code: str
    track_ids: tuple[str, ...]
    paths: tuple[Path, ...] = ()
    detail: str = ""


class PlanRejected(ValueError):
    def __init__(self, issues: tuple[PlanIssue, ...]):
        self.issues = issues
        super().__init__("; ".join(f"{i.code}: {i.detail}" for i in issues))


@dataclass(frozen=True)
class PlannedWork:
    work_id: str
    kind: str
    relative_path: str
    track_ids: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateGroup:
    group_id: str
    work_id: str
    members: tuple[str, ...]
    representative_track_id: str


@dataclass(frozen=True)
class OutputUse:
    path: Path
    role: str
    track_id: str
    owner_id: str
    access: str


@dataclass(frozen=True)
class PlannedTrack:
    track_id: str
    work_id: str
    ordinal: int
    task: AudioTask
    source_stamp: tuple
    dependencies: tuple[str, ...] = ()
    representative_track_id: str | None = None
    verification: str = "fresh"


@dataclass(frozen=True)
class Provenance:
    track_id: str
    source_stamp: tuple
    settings_hash: str
    output_key: str
    output_hash: str
    output_type: str


@dataclass(frozen=True)
class ExecutionPlan:
    root: Path
    root_id: str
    plan_id: str
    plan_hash: str
    mode: str
    settings_json: str
    settings_hash: str
    works: tuple[PlannedWork, ...]
    tracks: tuple[PlannedTrack, ...]
    duplicate_groups: tuple[DuplicateGroup, ...]
    selected_work_ids: tuple[str, ...]
    excluded_track_ids: tuple[str, ...]
    issues: tuple[PlanIssue, ...]
    inventory: tuple[tuple[Path, tuple], ...]
    source_inventory_fingerprint: str

    @property
    def settings(self) -> dict:
        return json.loads(self.settings_json)

    @cached_property
    def planned_tracks(self) -> tuple[PlannedTrack, ...]:
        selected, excluded = set(self.selected_work_ids), set(self.excluded_track_ids)
        return tuple(t for t in self.tracks if t.work_id in selected and t.track_id not in excluded)

    @cached_property
    def track_index(self) -> Mapping[str, PlannedTrack]:
        return MappingProxyType({t.track_id: t for t in self.tracks})

    @cached_property
    def output_ownership(self) -> tuple[OutputUse, ...]:
        return _namespace_uses(self.planned_tracks, self.duplicate_groups, self.mode)

    @property
    def executable(self) -> bool:
        return not self.issues

    def assert_ready(self) -> None:
        if self.issues:
            raise PlanRejected(self.issues)
        changed = tuple(path for path, stamp in self.inventory if file_stamp(path) != stamp)
        if changed:
            raise PlanRejected((PlanIssue("PLAN_STALE", (), changed, "inventory changed after planning"),))


def make_provenance(plan: ExecutionPlan, track_id: str, output: Path, output_type: str) -> Provenance:
    """Receipt constructor for a caller that KNOWS a stage completed.

    This does not certify completeness of arbitrary legacy files; import/migration
    policy is outside Phase 1A. Safe execution uses it only after stage validation.
    """
    track = plan.track_index[track_id]
    if output_type not in dict(_output_paths(track.task)) or path_key(output) != path_key(dict(_output_paths(track.task))[output_type]):
        raise ValueError("receipt output is not owned by this Track/stage")
    validate_srt(output)
    return Provenance(track_id, track.source_stamp, plan.settings_hash, path_key(output),
                      hashlib.sha256(output.read_bytes()).hexdigest(), output_type)


def _verified(track: PlannedTrack, path: Path, kind: str, settings_hash: str,
              provenance: frozenset[Provenance]) -> bool:
    try:
        validate_srt(path)
        expected = Provenance(track.track_id, track.source_stamp, settings_hash, path_key(path),
                              hashlib.sha256(path.read_bytes()).hexdigest(), kind)
        return expected in provenance
    except (OSError, ValueError):
        return False


def _output_paths(task: AudioTask) -> tuple[tuple[str, Path], ...]:
    return (("final", task.final_srt_path), ("transcription", task.transcribe_srt_path),
            ("translation", task.translate_srt_path), ("opencc", task.opencc_srt_path))


def _namespace_uses(tracks: tuple[PlannedTrack, ...], groups: tuple[DuplicateGroup, ...], mode: str) -> tuple[OutputUse, ...]:
    by_id = {t.track_id: t for t in tracks}
    aliases = {}
    for group in groups:
        representative = by_id.get(group.representative_track_id)
        if representative:
            rep_paths = dict(_output_paths(representative.task))
            for member in group.members:
                if member in by_id:
                    for role, path in _output_paths(by_id[member].task):
                        if path_key(path) == path_key(rep_paths[role]):
                            aliases[(member, role)] = group.representative_track_id
    uses = []
    for track in tracks:
        if is_existing_subtitle_skip(track.task):
            source = track.task.existing_subtitle_path
            if source:
                uses.append(OutputUse(source, "existing_subtitle", track.track_id, track.track_id, "read"))
            uses.append(OutputUse(track.task.audio_path, "source_audio", track.track_id, track.track_id, "read"))
            continue
        for role, path in _output_paths(track.task):
            action = track.task.action
            mode_role = "transcription" if mode == "transcribe_only" else "final"
            access = "write"
            if action == "skip":
                if role != "final":
                    continue
                access = "read"
            elif action in {"copy_existing_subtitle", "copy_duplicate_audio_subtitle"}:
                if role != mode_role:
                    continue
                source = track.task.existing_subtitle_path
                if path.exists() or (source and path_key(source) == path_key(path)):
                    access = "read"
            elif mode == "transcribe_only" and role == "final":
                # Legacy process_audio_task reads final.exists() in BOTH modes.
                access = "read"
            # Transcribe-only can DELETE translate/opencc intermediates, so those
            # potential writes remain reserved even when currently absent.
            for suffix in ["", ".tmp"]:
                if suffix and access == "read":
                    continue
                target = path.with_name(path.name + suffix)
                owner = aliases.get((track.track_id, role), track.track_id)
                uses.append(OutputUse(target, role + suffix, track.track_id, owner, access))
        source = track.task.existing_subtitle_path
        if source:
            owner = track.dependencies[0] if track.dependencies else track.track_id
            uses.append(OutputUse(source, "transcription" if mode == "transcribe_only" else "final",
                                  track.track_id, owner, "read"))
        uses.append(OutputUse(track.task.audio_path, "source_audio", track.track_id, track.track_id, "read"))
    return tuple(uses)


def _namespace_issues(tracks: tuple[PlannedTrack, ...], groups: tuple[DuplicateGroup, ...], mode: str) -> list[PlanIssue]:
    uses: dict[object, list[OutputUse]] = {}
    for use in _namespace_uses(tracks, groups, mode):
        uses.setdefault(("path", path_key(use.path)), []).append(use)
        if use.path.exists():
            stat = use.path.stat()
            uses.setdefault(("inode", stat.st_dev, stat.st_ino), []).append(use)
    issues = []
    seen = set()
    for entries in uses.values():
        roles_owners = {(use.role, use.owner_id) for use in entries}
        if len(roles_owners) > 1 and any(use.access == "write" for use in entries):
            ids = tuple(sorted({use.track_id for use in entries}))
            paths = tuple(sorted({use.path for use in entries}, key=str))
            key = (ids, paths)
            if key not in seen:
                issues.append(PlanIssue("OUTPUT_NAMESPACE_COLLISION", ids, paths, "conflicting output owners/stages"))
                seen.add(key)
    return issues


def build_execution_plan(root: Path | str, *, mode: str = "full",
                         selected_work_ids: Iterable[str] | None = None,
                         limit: int | None = None, settings: Mapping | None = None,
                         provenance: Iterable[Provenance] = (),
                         skip_existing_subtitles: bool = False) -> ExecutionPlan:
    if mode not in {"full", "transcribe_only"}:
        raise ValueError("unsupported execution mode")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if not isinstance(skip_existing_subtitles, bool):
        raise ValueError("skip_existing_subtitles must be boolean")
    root = Path(root).expanduser().resolve()
    tasks = discover_audio_tasks(root)
    # Import only at call time: CLI owns defaults; it does not import this planner.
    from .cli import FAST_CHUNK_LENGTH, _effective_initial_prompt, _effective_transcribe_model, build_parser
    parser = build_parser()
    effective = vars(parser.parse_args(["--path", str(root)]))
    actions = {action.dest: action for action in parser._actions}
    for key, value in (settings or {}).items():
        if key not in effective or key in {"path", "execute", "dry_run", "limit", "transcribe_only"}:
            raise ValueError(f"unsupported plan setting: {key}")
        action = actions[key]
        if isinstance(effective[key], bool) and not isinstance(value, bool):
            raise ValueError(f"{key} must be boolean")
        if action.type is int and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{key} must be an integer")
        if action.choices and value not in action.choices:
            raise ValueError(f"unsupported {key}: {value}")
        if action.type is None and not isinstance(effective[key], bool) and not isinstance(value, str) and not (value is None and effective[key] is None):
            raise ValueError(f"{key} must be text")
        effective[key] = value
    for key in ["parallel", "transcribe_parallel", "transcribe_batch_size", "translate_chunk_size", "kotoba_chunk_length"]:
        if effective[key] < 1:
            raise ValueError(f"{key} must be >= 1")
    effective["transcribe_model"] = _effective_transcribe_model(effective["transcribe_backend"], effective["transcribe_model"])
    effective["initial_prompt"] = _effective_initial_prompt(effective["initial_prompt"], effective["no_asmr_prompt"])
    effective["xxl_chunk_length"] = FAST_CHUNK_LENGTH if effective["fast_transcribe"] else None
    effective.update(transcribe_only=mode == "transcribe_only", limit=limit)
    if skip_existing_subtitles:
        effective["gui_existing_subtitles"] = "skip_common"
    settings_json = json.dumps(effective, sort_keys=True, ensure_ascii=False)
    settings_hash = _hash(effective)
    root_id = str(uuid5(APP_NAMESPACE, path_key(root)))
    directories = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: (p.name.casefold(), p.name))
    work_paths = [(p, "directory", p.name) for p in directories]
    if any(t.work_dir == root for t in tasks):
        work_paths.append((root, "synthetic_root", ""))
    work_ids = {p: str(uuid5(UUID(root_id), "synthetic-root" if kind == "synthetic_root" else "work/" + name))
                for p, kind, name in work_paths}
    tracks = tuple(PlannedTrack(str(uuid5(UUID(root_id), "track/" + t.relative_audio_path.as_posix())),
                                work_ids[t.work_dir], index, t, file_stamp(t.audio_path))
                   for index, t in enumerate(tasks))
    tracks_by_work: dict[Path, list[PlannedTrack]] = {}
    for track in tracks:
        tracks_by_work.setdefault(track.task.work_dir, []).append(track)
    groups = []
    for work_dir, kind, _ in work_paths:
        work_tracks = tracks_by_work.get(work_dir, [])
        # Root-level discovery intentionally retains the legacy no-dedupe behavior.
        raw_groups = _group_same_content_tasks([t.task for t in work_tracks]) if kind == "directory" else [[t.task] for t in work_tracks]
        by_path = {t.task.audio_path: t for t in work_tracks}
        for group in raw_groups:
            representative = _choose_representative_audio(group, work_dir)
            ids = tuple(by_path[t.audio_path].track_id for t in group)
            groups.append(DuplicateGroup(str(uuid5(UUID(work_ids[work_dir]), "group/" + "|".join(ids))),
                                         work_ids[work_dir], ids, by_path[representative.audio_path].track_id))
    by_id = {t.track_id: t for t in tracks}
    final_owners = {path_key(t.task.final_srt_path): t.track_id for t in tracks}
    group_by_member = {member: group for group in groups for member in group.members}
    adjusted = []
    for track in tracks:
        group = group_by_member[track.track_id]
        task = track.task
        dependencies = ()
        existing = _existing_common_subtitle(task) if skip_existing_subtitles else None
        if (skip_existing_subtitles and existing is None and task.action == "copy_existing_subtitle"
                and task.existing_subtitle_path and task.existing_subtitle_path.suffix.lower() != ".srt"):
            existing = task.existing_subtitle_path
        if existing is not None:
            task = replace(task, action="skip", status="skipped", existing_subtitle_path=existing,
                           reason="existing_subtitle_skip")
        elif task.action == "copy_duplicate_audio_subtitle":
            dependencies = (group.representative_track_id,)
            if mode == "transcribe_only":
                task = replace(task, existing_subtitle_path=by_id[dependencies[0]].task.transcribe_srt_path)
        elif task.action == "copy_existing_subtitle" and task.existing_subtitle_path:
            owner = final_owners.get(path_key(task.existing_subtitle_path))
            if owner and owner != track.track_id:
                dependencies = (owner,)
                if mode == "transcribe_only" and not (skip_existing_subtitles and by_id[owner].task.action == "skip"):
                    task = replace(task, existing_subtitle_path=by_id[owner].task.transcribe_srt_path)
        adjusted.append(replace(track, task=task, dependencies=dependencies,
                                representative_track_id=group.representative_track_id))
    tracks = tuple(adjusted)
    works = tuple(PlannedWork(work_ids[p], kind, name, tuple(t.track_id for t in tracks_by_work.get(p, [])))
                  for p, kind, name in work_paths)
    selected = tuple(w.work_id for w in works) if selected_work_ids is None else tuple(selected_work_ids)
    selected_set = set(selected)
    if len(selected_set) != len(selected) or selected_set - {w.work_id for w in works}:
        raise ValueError("unknown or repeated Work selection")
    selected = tuple(w.work_id for w in works if w.work_id in selected_set)
    selected_tracks = tuple(t for t in tracks if t.work_id in selected_set)
    processes = [t.track_id for t in selected_tracks if t.task.action == "process"]
    excluded = tuple(processes[limit:]) if limit is not None else ()
    excluded_set = set(excluded)
    planned_ids = {t.track_id for t in selected_tracks if t.track_id not in excluded_set}
    selected_ids = {t.track_id for t in selected_tracks}
    issues = [issue for issue in _namespace_issues(tracks, tuple(groups), mode) if selected_ids.intersection(issue.track_ids)]
    if not planned_ids:
        issues.append(PlanIssue("EMPTY_EXECUTION_PLAN", (), detail="no selected planned Tracks"))
    proof = frozenset(provenance)
    by_id = {t.track_id: t for t in tracks}
    verified_tracks = []
    for track in tracks:
        verification = "unverified_legacy" if is_existing_subtitle_skip(track.task) else "fresh"
        if track.track_id in planned_ids:
            task = track.task
            if any(dep in excluded_set for dep in track.dependencies):
                issues.append(PlanIssue("DEPENDENCY_EXCLUDED", (track.track_id,) + track.dependencies,
                                        detail="limit excludes required representative"))
            source = task.existing_subtitle_path
            if is_existing_subtitle_skip(task):
                if (not source or not source.is_file()
                        or not source.resolve().is_relative_to(task.work_dir.resolve())
                        or not source.resolve().is_relative_to(root)):
                    issues.append(PlanIssue("INVALID_EXISTING_SUBTITLE", (track.track_id,),
                                            (source,) if source else (), "existing subtitle is not a file within the Work"))
            for dependency in track.dependencies:
                owner = by_id[dependency]
                if owner.task.action == "skip" and source and not source.exists():
                    issues.append(PlanIssue("DEPENDENCY_OUTPUT_UNAVAILABLE", (track.track_id, dependency), (source,), "skipped representative lacks the mode-specific output"))
            if task.action == "copy_existing_subtitle" and source and source.suffix.lower() != ".srt":
                issues.append(PlanIssue("UNSUPPORTED_SUBTITLE_CONVERSION", (track.track_id,), (source,), "legacy raw alternative conversion is not GUI-safe"))
            if task.action == "skip":
                verification = "unverified_legacy" if is_existing_subtitle_skip(task) else (
                    "verified" if _verified(track, task.final_srt_path, "final", settings_hash, proof) else "unverified_legacy")
            elif task.action == "process":
                for role, path in _output_paths(task):
                    if path.exists() and not _verified(track, path, role, settings_hash, proof):
                        verification = "unverified_legacy"
                        issues.append(PlanIssue("UNVERIFIED_LEGACY", (track.track_id,), (path,), "no matching completed-stage provenance"))
                    elif path.exists():
                        verification = "verified"
                    tmp = path.with_name(path.name + ".tmp")
                    if tmp.exists():
                        issues.append(PlanIssue("UNVERIFIED_LEGACY", (track.track_id,), (tmp,), "uncommitted temporary output"))
            elif source and source.exists():
                owner = by_id[track.dependencies[0]] if track.dependencies else track
                kind = "transcription" if mode == "transcribe_only" else "final"
                if skip_existing_subtitles and path_key(source) == path_key(owner.task.final_srt_path):
                    kind = "final"
                if source.suffix.lower() == ".srt" and not _verified(owner, source, kind, settings_hash, proof):
                    issues.append(PlanIssue("UNVERIFIED_LEGACY", (track.track_id,), (source,), "copy source lacks matching provenance"))
            if task.action in {"copy_existing_subtitle", "copy_duplicate_audio_subtitle"}:
                target = task.transcribe_srt_path if mode == "transcribe_only" else task.final_srt_path
                kind = "transcription" if mode == "transcribe_only" else "final"
                if target.exists() and not _verified(track, target, kind, settings_hash, proof):
                    verification = "unverified_legacy"
                    issues.append(PlanIssue("UNVERIFIED_LEGACY", (track.track_id,), (target,), "copy target lacks matching provenance"))
        verified_tracks.append(replace(track, verification=verification))
    tracks = tuple(verified_tracks)
    inventory_paths = {t.task.audio_path for t in selected_tracks}
    for track in selected_tracks:
        for _, path in _output_paths(track.task):
            inventory_paths.update([path, path.with_name(path.name + ".tmp")])
        if track.task.existing_subtitle_path:
            inventory_paths.add(track.task.existing_subtitle_path)
    inventory = tuple((p, file_stamp(p)) for p in sorted(inventory_paths, key=str))
    inventory_hash = _hash([(t.track_id, t.source_stamp) for t in selected_tracks])
    plan_hash = _hash({"settings": effective, "selected": selected, "excluded": excluded,
                       "tracks": [(t.track_id, t.task.action, t.dependencies, t.verification) for t in tracks],
                       **({"existing_subtitles": [(t.track_id, path_key(t.task.existing_subtitle_path))
                                                  for t in tracks if is_existing_subtitle_skip(t.task)]}
                          if skip_existing_subtitles else {}),
                       "groups": [(g.group_id, g.members, g.representative_track_id) for g in groups],
                       "issues": [(i.code, i.track_ids, [str(p) for p in i.paths]) for i in issues],
                       "inventory": [(str(p), stamp) for p, stamp in inventory]})
    return ExecutionPlan(root, root_id, str(uuid5(UUID(root_id), "plan/" + plan_hash)), plan_hash,
                         mode, settings_json, settings_hash, works, tracks, tuple(groups), selected,
                         excluded, tuple(issues), inventory, inventory_hash)
