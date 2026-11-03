"""Exercise pytest path/cleanup policy in isolated child pytest sessions."""
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def run_isolated_pytest(tmp_path, body, *options):
    project = tmp_path / "project"
    suite = project / "tests"
    suite.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("conftest.py"), suite / "conftest.py")
    (suite / "test_sample.py").write_text(body, encoding="utf-8")
    temp = tmp_path / "external-temp"
    temp.mkdir()
    env = dict(os.environ, TEMP=str(temp), TMP=str(temp), TMPDIR=str(temp),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run([sys.executable, "-B", "-m", "pytest", "tests", "-q", "-s", *options],
        cwd=project, env=env, text=True, capture_output=True, timeout=30)
    return project, temp, result


@pytest.mark.parametrize("fail", [False, True])
def test_external_pytest_artifacts_cleanup_and_failure_evidence(tmp_path, fail):
    body = '''import json
def test_sample(tmp_path, pytestconfig):
    print("OBSERVED=" + json.dumps({"tmp": str(tmp_path), "cache": pytestconfig.getini("cache_dir")}))
    (tmp_path / "evidence.txt").write_text("owned synthetic evidence")
    assert FAIL is False
'''.replace("FAIL", repr(fail))
    project, temp, result = run_isolated_pytest(tmp_path, body, "-o", "cache_dir=.pytest_cache")
    assert result.returncode == int(fail), result.stdout + result.stderr
    observed = json.loads(next(line.removeprefix("OBSERVED=")
        for line in result.stdout.splitlines() if line.startswith("OBSERVED=")))
    artifact_root = temp / "VoiceTransl" / "Tests"
    data = Path(observed["tmp"])
    cache = Path(observed["cache"])
    assert data.is_relative_to(artifact_root)
    assert cache.parent.parent == artifact_root
    assert data.is_relative_to(cache.parent)
    assert not (project / ".pytest_cache").exists()
    if fail:
        assert "pytest evidence retained (exit 1):" in result.stdout
        assert (data / "evidence.txt").read_text() == "owned synthetic evidence"
        assert (cache.parent / "owner.json").exists()
        assert cache.exists()
    else:
        assert "disposable pytest artifacts cleaned:" in result.stdout
        assert not cache.parent.exists()


def test_repository_basetemp_rejected_without_deleting_historical_data(tmp_path):
    # The existing directory belongs to the caller, never to the pytest run.
    historical = tmp_path / "project" / ".asmr_historical"
    historical.mkdir(parents=True)
    sentinel = historical / "keep.txt"
    sentinel.write_text("caller evidence", encoding="utf-8")
    project, temp, result = run_isolated_pytest(tmp_path, "def test_sample(tmp_path): pass\n",
        "--basetemp", str(historical))
    assert result.returncode == 4
    assert "--basetemp must be outside the repository" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "caller evidence"
    assert not (project / ".pytest_cache").exists()
    assert not (temp / "VoiceTransl" / "Tests").exists()


def test_real_smoke_defaults_external_and_requires_prepared_ownership(tmp_path, monkeypatch):
    path = Path(__file__).parent / "fixtures" / "asmr_gui_real_smoke.py"
    spec = importlib.util.spec_from_file_location("hygiene_real_smoke", path)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    root = tmp_path / "external-smoke-root"
    monkeypatch.setattr(smoke, "ARTIFACT_ROOT", root)
    prepared = []
    monkeypatch.setattr(smoke, "prepare", lambda base: prepared.append(base) or {
        "settings_override": {}, "audio_inventory": [], "sakura": {}})
    monkeypatch.setattr(sys, "argv", [str(path), "--prepare"])
    smoke.main()
    assert prepared[0].parent == root
    assert prepared[0].name.startswith("real-smoke-")
    assert not prepared[0].is_relative_to(smoke.PROJECT)
    monkeypatch.setattr(sys, "argv", [str(path), "--base", str(prepared[0]),
        "--baseline-green", "--scenario", "full"])
    with pytest.raises(SystemExit) as missing_owner:
        smoke.main()
    assert missing_owner.value.code == 2
    monkeypatch.setattr(sys, "argv", [str(path), "--prepare", "--base",
        str(smoke.PROJECT / ".asmr_phase1c_real_unowned")])
    with pytest.raises(SystemExit) as repository_base:
        smoke.main()
    assert repository_base.value.code == 2
    assert len(prepared) == 1  # No generation, inference, or repository write.
