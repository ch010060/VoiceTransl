"""Post-receipt cleanup of exact, freshly validated GUI-owned subtitle paths."""
from contextlib import contextmanager
import os
from pathlib import Path
import stat

from .gui_plan import file_stamp, path_key

ROLES = ("transcription", "translation", "opencc")
WARNING = "INTERMEDIATE_CLEANUP_INCOMPLETE"


def _paths(task):
    return (task.transcribe_srt_path, task.translate_srt_path, task.opencc_srt_path)


def _plain(path, directory=False):
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
        raise ValueError("unsafe path type")
    return info


def _directories(path):
    identities = []
    for parent in (*reversed(path.parent.parents), path.parent):
        info = _plain(parent, True)
        identities.append((str(parent), info.st_dev, info.st_ino))
    return tuple(identities)


@contextmanager
def _pinned_parent(path, expected):
    handles = []
    descriptor = None
    kernel = None
    try:
        if os.name == "nt":
            import ctypes
            from .gui_ownership import _kernel
            kernel = _kernel()
            # No FILE_SHARE_DELETE: prevent directory rename/reparse substitution.
            for name, _, _ in expected:
                handle = kernel.CreateFileW(name, 0, 3, None, 3, 0x02200000, None)
                if handle == ctypes.c_void_p(-1).value:
                    raise OSError("cannot pin output directory")
                handles.append(handle)
        else:
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if _directories(path) != expected:
            raise ValueError("output directory changed")
        if descriptor is not None:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != expected[-1][1:]:
                raise ValueError("output directory changed")
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for handle in reversed(handles):
            kernel.CloseHandle(handle)


class IntermediateCleanup:
    def __init__(self, plan, ownership_check):
        self.plan, self.ownership_check = plan, ownership_check
        self.directories, self.stamps = {}, {}
        for track in plan.planned_tracks:
            for path in _paths(track.task):
                try:
                    self.directories[str(path)] = _directories(path)
                except (OSError, ValueError):
                    pass  # Unsafe/missing ancestors are never eligible for deletion.

    def capture(self, track_id, path):
        """Called only by the trusted stage validator, never by a worker message."""
        track = self.plan.track_index[track_id]
        if path not in _paths(track.task):
            return
        try:
            _plain(path)
            if _directories(path) != self.directories.get(str(path)):
                return
            self.stamps[(track_id, str(path))] = file_stamp(path)
        except (OSError, ValueError):
            pass

    def run(self, results, *, interrupted=False):
        records = {}
        for result in results:
            track = self.plan.track_index[result.track_id]
            if (result.status != "success" or self.plan.mode != "full"
                    or track.task.action != "process" or track.shared_output_owner):
                continue
            record = {"status": "disabled", "files": {}}
            records[track.track_id] = record
            if self.plan.settings.get("keep_intermediate_files", False):
                continue
            if interrupted:
                record["status"] = "deferred_interrupted"
                continue
            for role, path in zip(ROLES, _paths(track.task)):
                uses = [u for u in self.plan.output_ownership if u.path == path]
                if any(u.access == "read" and u.track_id != track.track_id for u in uses):
                    record["files"][role] = "deferred_dependencies"
                    continue
                try:
                    if self.ownership_check is None:
                        raise ValueError("ownership unavailable")
                    self.ownership_check()
                    if not any(u.track_id == track.track_id and u.owner_id == track.track_id
                               and u.role == role and u.access == "write" for u in uses):
                        raise ValueError("not owned")
                    if (not path.is_relative_to(self.plan.root) or path == self.plan.root
                            or any(path_key(path) == path_key(t.task.final_srt_path)
                                   or (t.task.existing_subtitle_path is not None
                                       and path_key(path) == path_key(t.task.existing_subtitle_path))
                                   for t in self.plan.planned_tracks)):
                        raise ValueError("protected output")
                    expected = self.stamps.get((track.track_id, str(path)))
                    if expected is None:
                        raise ValueError("no validated artifact")
                    with _pinned_parent(path, self.directories[str(path)]) as descriptor:
                        _plain(path)
                        if file_stamp(path) != expected:
                            raise ValueError("artifact changed")
                        self.ownership_check()
                        # dir_fd keeps POSIX unlink anchored to the checked directory.
                        if descriptor is None:
                            path.unlink()
                        else:
                            os.unlink(path.name, dir_fd=descriptor)
                    record["files"][role] = "removed"
                except (OSError, ValueError, KeyError, RuntimeError):
                    # Bounded codes only: never publish filesystem paths or subtitle text.
                    record["files"][role] = "failed_or_unsafe"
            statuses = set(record["files"].values())
            record["status"] = ("completed" if statuses == {"removed"} else
                                "deferred_dependencies" if statuses == {"deferred_dependencies"} else
                                "partial_failed")
        return records
