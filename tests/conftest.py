"""Keep disposable pytest data outside the checkout and preserve failed runs."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import uuid

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    project = Path(__file__).resolve().parents[1]
    explicit = config.option.basetemp
    if explicit and Path(explicit).resolve().is_relative_to(project):
        raise pytest.UsageError(
            "--basetemp must be outside the repository; omit it to use "
            "%TEMP%/VoiceTransl/Tests. Historical artifacts are never removed."
        )
    parent = (Path(tempfile.gettempdir()) / "VoiceTransl" / "Tests").resolve()
    if parent.is_relative_to(project):
        raise pytest.UsageError("TEMP must point outside the repository for disposable pytest artifacts")
    parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    owned = parent / ("pytest-" + token)
    owned.mkdir()
    marker = {"kind": "VoiceTransl pytest", "token": token, "path": str(owned)}
    (owned / "owner.json").write_text(json.dumps(marker), encoding="utf-8")
    config._voicetransl_artifacts = (parent, owned, marker)
    if not explicit:
        config.option.basetemp = str(owned / "tmp")
    # This hook runs before pytest's cache and tmp_path factories initialize.
    config.inicfg["cache_dir"] = str(owned / "cache")
    config.option.override_ini = [*(config.option.override_ini or []),
        "cache_dir=" + str(owned / "cache")]


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    parent, owned, marker = session.config._voicetransl_artifacts
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if exitstatus == 0:
        try:
            # Never traverse or remove a caller-selected directory, link, or
            # historical path. Only the exact marked UUID created above is ours.
            if owned.is_symlink() or owned.resolve() != owned or owned.parent != parent:
                raise OSError("owned artifact directory identity changed")
            if json.loads((owned / "owner.json").read_text(encoding="utf-8")) != marker:
                raise OSError("owned artifact marker changed")
            shutil.rmtree(owned)
        except (OSError, ValueError) as exc:
            message = f"VoiceTransl cleanup incomplete; artifacts retained: {owned} ({exc})"
        else:
            message = f"VoiceTransl disposable pytest artifacts cleaned: {owned}"
    else:
        message = f"VoiceTransl pytest evidence retained (exit {exitstatus}): {owned}"
        if not Path(session.config.option.basetemp).resolve().is_relative_to(owned):
            message += f"; caller-owned --basetemp: {session.config.option.basetemp}"
    if reporter:
        reporter.write_line(message)
