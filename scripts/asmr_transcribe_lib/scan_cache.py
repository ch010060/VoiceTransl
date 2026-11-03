"""Canonical identities reused only inside one synchronous validation phase.

No exists/stat/hash cache and no persistent state. Every phase is thread-local;
membership traversals explicitly start an independent fresh phase.
"""
from contextvars import ContextVar
from functools import wraps

_active = ContextVar("asmr_scan_phase", default=None)


class _Phase:
    def __init__(self):
        self.canonical = {}
        self.subtitles = {}
        self.groups = {}


def current_phase():
    return _active.get()


def scan_phase(function=None, *, fresh=False):
    if function is None:
        return lambda target: scan_phase(target, fresh=fresh)

    @wraps(function)
    def scoped(*args, **kwargs):
        if not fresh and _active.get() is not None:
            return function(*args, **kwargs)
        token = _active.set(_Phase())
        try:
            return function(*args, **kwargs)
        finally:
            _active.reset(token)

    return scoped


def resolved_path(path):
    phase = _active.get()
    if phase is None:
        return path.resolve()
    if path not in phase.canonical:
        # Existing entries, including dangling symlinks and junctions, always
        # use realpath. For an ordinary missing leaf, Windows non-strict
        # realpath appends it to its resolved ancestor. Reuse that actual
        # ancestor rather than reopening it for every absent output filename.
        missing = False
        if (path.is_absolute() and path.name and ".." not in path.parts
                and not path.name.endswith((".", " ")) and not path.is_reserved()):
            try:
                path.lstat()
            except FileNotFoundError:
                missing = True
            except OSError:
                pass  # Preserve pathlib's own error/normalization behavior.
        phase.canonical[path] = resolved_path(path.parent) / path.name if missing else path.resolve()
    return phase.canonical[path]
