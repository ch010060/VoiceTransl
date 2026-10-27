from __future__ import annotations

from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
import re
import unicodedata

from .models import AudioTask

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav"}
ALTERNATE_SUBTITLE_EXTENSIONS = (".vtt", ".ass", ".ssa", ".lrc")
AUDIO_REPRESENTATIVE_PRIORITY = {".flac": 0, ".wav": 1, ".mp3": 2}


def discover_audio_tasks(root: Path | str) -> list[AudioTask]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"root path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"root path is not a directory: {root_path}")

    tasks: list[AudioTask] = []

    work_dirs = sorted(
        [child for child in root_path.iterdir() if child.is_dir()],
        key=lambda path: path.name.casefold(),
    )
    for work_dir in work_dirs:
        tasks.extend(_discover_work_tasks(root_path, work_dir, work_dir.name))

    root_audio_files = sorted(
        [child for child in root_path.iterdir() if child.is_file() and _is_audio(child)],
        key=lambda path: path.name.casefold(),
    )
    for audio_path in root_audio_files:
        tasks.append(_build_task(root_path, root_path, "_root", audio_path))

    return tasks


def _discover_work_tasks(root_path: Path, work_dir: Path, work_name: str) -> list[AudioTask]:
    audio_files = sorted(
        [path for path in work_dir.rglob("*") if path.is_file() and _is_audio(path)],
        key=lambda path: path.relative_to(work_dir).as_posix().casefold(),
    )
    tasks = [_build_task(root_path, work_dir, work_name, audio_path) for audio_path in audio_files]
    return _dedupe_same_stem_audio(tasks, work_dir)


def _dedupe_same_stem_audio(tasks: list[AudioTask], work_dir: Path) -> list[AudioTask]:
    grouped = _group_same_content_tasks(tasks)

    replacements: dict[Path, AudioTask] = {}
    for duplicate_tasks in grouped:
        if len(duplicate_tasks) < 2:
            continue

        existing_subtitle = _best_existing_subtitle_source(duplicate_tasks)
        if existing_subtitle is not None:
            for task in duplicate_tasks:
                if task.final_srt_path.exists():
                    continue
                replacements[task.audio_path] = replace(
                    task,
                    status="skipped",
                    action="copy_existing_subtitle",
                    reason=f"existing_duplicate_subtitle:{existing_subtitle.suffix.lower()}",
                    existing_subtitle_path=existing_subtitle,
                )
            continue

        representative = _choose_representative_audio(duplicate_tasks, work_dir)
        for task in duplicate_tasks:
            if task.audio_path == representative.audio_path:
                continue
            replacements[task.audio_path] = replace(
                task,
                status="skipped",
                action="copy_duplicate_audio_subtitle",
                reason=f"duplicate_audio:{representative.audio_path.relative_to(work_dir).as_posix()}",
                existing_subtitle_path=representative.final_srt_path,
            )

    return [replacements.get(task.audio_path, task) for task in tasks]


def _group_same_content_tasks(tasks: list[AudioTask]) -> list[list[AudioTask]]:
    if not tasks:
        return []

    keys = [_content_stem_key(task.audio_path.stem) for task in tasks]
    parents = list(range(len(tasks)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(tasks)):
        for right in range(left + 1, len(tasks)):
            if _same_content_task(tasks[left], keys[left], tasks[right], keys[right]):
                union(left, right)

    grouped: dict[int, list[AudioTask]] = {}
    for index, task in enumerate(tasks):
        grouped.setdefault(find(index), []).append(task)
    return list(grouped.values())


def _same_content_task(left_task: AudioTask, left_key: str, right_task: AudioTask, right_key: str) -> bool:
    if left_key == right_key:
        return True
    if left_task.audio_path.parent == right_task.audio_path.parent:
        return False
    return _same_content_stem(left_key, right_key)


def _best_existing_subtitle_source(tasks: list[AudioTask]) -> Path | None:
    candidates: list[Path] = []
    for task in tasks:
        if task.final_srt_path.exists():
            candidates.append(task.final_srt_path)
        elif task.existing_subtitle_path is not None and task.existing_subtitle_path.exists():
            candidates.append(task.existing_subtitle_path)
    if not candidates:
        return None
    return min(candidates, key=lambda path: (_subtitle_priority(path), path.as_posix().casefold()))


def _choose_representative_audio(tasks: list[AudioTask], work_dir: Path) -> AudioTask:
    return min(
        tasks,
        key=lambda task: (
            AUDIO_REPRESENTATIVE_PRIORITY.get(task.audio_path.suffix.lower(), 99),
            task.audio_path.relative_to(work_dir).as_posix().casefold(),
        ),
    )


def _subtitle_priority(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return 0
    try:
        return 1 + ALTERNATE_SUBTITLE_EXTENSIONS.index(suffix)
    except ValueError:
        return 99


def _content_stem_key(stem: str) -> str:
    normalized = unicodedata.normalize("NFKC", stem).casefold()
    normalized = normalized.replace("ＳＥ", "se")
    normalized = re.sub(r"^[\s_\-－–—]*[【\[\(（]?\s*se\s*(?:なし|無し|ナシ|無)[】\]\)）]?[\s_\-－–—]*", "", normalized)
    normalized = re.sub(r"[\s_\-－–—]*[【\[\(（]\s*se\s*(?:なし|無し|ナシ|無)\s*[】\]\)）]\s*$", "", normalized)
    normalized = re.sub(r"[\s_\-－–—]+se\s*(?:なし|無し|ナシ|無)\s*$", "", normalized)
    return normalized.strip(" _-－–—")


def _same_content_stem(left: str, right: str) -> bool:
    if left == right:
        return True

    left_comp = _comparison_stem(left)
    right_comp = _comparison_stem(right)
    if left_comp == right_comp:
        return True
    if min(len(left_comp), len(right_comp)) < 8:
        return False

    left_numbers = re.findall(r"\d+", left_comp)
    right_numbers = re.findall(r"\d+", right_comp)
    if left_numbers and right_numbers and left_numbers != right_numbers:
        return False

    ratio = SequenceMatcher(None, left_comp, right_comp).ratio()
    return ratio >= 0.88


def _comparison_stem(stem: str) -> str:
    normalized = unicodedata.normalize("NFKC", stem).casefold()
    return re.sub(r"[\s_\-－–—【】\[\]\(\)（）「」『』・,，.。!！?？:：;；'\"“”‘’&＆/\\]+", "", normalized)


def _build_task(root_path: Path, work_dir: Path, work_name: str, audio_path: Path) -> AudioTask:
    final_srt_path = audio_path.with_suffix(".srt")
    transcribe_srt_path = audio_path.with_name(f"{audio_path.stem}-transcribe.srt")
    translate_srt_path = audio_path.with_name(f"{audio_path.stem}-translate.srt")
    opencc_srt_path = audio_path.with_name(f"{audio_path.stem}-opencc.srt")

    status = "queued"
    action = "process"
    reason: str | None = None
    existing_subtitle_path: Path | None = None

    if final_srt_path.exists():
        status = "skipped"
        action = "skip"
        reason = "existing_srt"
    else:
        existing_subtitle_path = _find_existing_alternate_subtitle(audio_path)
        if existing_subtitle_path is not None:
            status = "skipped"
            action = "copy_existing_subtitle"
            reason = f"existing_alt_subtitle:{existing_subtitle_path.suffix.lower()}"

    return AudioTask(
        work_name=work_name,
        work_dir=work_dir,
        audio_path=audio_path,
        root_relative_audio_path=audio_path.relative_to(root_path),
        relative_audio_path=audio_path.relative_to(root_path),
        final_srt_path=final_srt_path,
        transcribe_srt_path=transcribe_srt_path,
        translate_srt_path=translate_srt_path,
        opencc_srt_path=opencc_srt_path,
        status=status,
        action=action,
        reason=reason,
        existing_subtitle_path=existing_subtitle_path,
    )


def _is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def _find_existing_alternate_subtitle(audio_path: Path) -> Path | None:
    for suffix in ALTERNATE_SUBTITLE_EXTENSIONS:
        candidate = audio_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
        upper_candidate = audio_path.with_suffix(suffix.upper())
        if upper_candidate.exists():
            return upper_candidate
    return None
