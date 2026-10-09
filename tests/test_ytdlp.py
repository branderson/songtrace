import os
import time

import pytest

from songtrace import ytdlp
from songtrace.config import Config
from songtrace.ytdlp import YtDlp, YtDlpError


def fake_package(path, version="2026.09.01", body=""):
    """A stand-in yt_dlp that answers --version and otherwise runs `body`."""
    pkg = path / "yt_dlp"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").touch()
    (pkg / "__main__.py").write_text(
        "import sys\n"
        f"if '--version' in sys.argv: print({version!r}); sys.exit()\n"
        + body)
    return path


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(ytdlp, "bundled", lambda: YtDlp("2026.8.19"))
    return ytdlp.cache_dir()


@pytest.fixture
def pypi(monkeypatch):
    """Stubs PyPI and pip: `latest` is the newest release, `installs` logs pip."""
    state = {"latest": "2026.9.1", "installs": []}
    monkeypatch.setattr(ytdlp, "_latest_release", lambda: state["latest"])
    def pip_install(target, version):
        state["installs"].append(version)
        fake_package(target, version)
    monkeypatch.setattr(ytdlp, "_pip_install", pip_install)
    return state


def cfg(version="auto"):
    return Config(whisper_url="http://x/v1", ytdlp_version=version)


def test_run_uses_the_install_on_its_path_and_logs_its_output(tmp_path):
    path = fake_package(tmp_path, body="print('[youtube] hello', sys.argv[1:])\n")
    logged = []
    YtDlp("2026.09.01", path).run(["--", "https://yt/x"], logged.append)
    assert logged == ["[youtube] hello ['--ignore-config', '--', 'https://yt/x']"]


def test_run_raises_the_error_line(tmp_path):
    path = fake_package(tmp_path, body=(
        "print('ERROR: HTTP Error 403: Forbidden', file=sys.stderr)\n"
        "print('trailing noise')\nsys.exit(1)\n"))
    with pytest.raises(YtDlpError, match="^ERROR: HTTP Error 403"):
        YtDlp("2026.09.01", path).run([], lambda m: None)


def test_run_flags_a_usage_error_as_incompatible(tmp_path):
    path = fake_package(tmp_path, body=(
        "print('error: no such option: --retries', file=sys.stderr)\n"
        "sys.exit(2)\n"))
    with pytest.raises(YtDlpError, match="2026.09.01 rejected.*--retries"):
        YtDlp("2026.09.01", path).run([], lambda m: None)


def test_auto_installs_a_newer_release_and_tries_it_first(cache, pypi):
    found = ytdlp.candidates(cfg(), lambda m: None)
    assert [y.version for y in found] == ["2026.9.1", "2026.8.19"]
    assert found[0].path == cache / "2026.9.1"
    assert not [p for p in cache.iterdir() if p.name.startswith(".tmp-")]


def test_auto_checks_pypi_once_a_day(cache, pypi):
    ytdlp.candidates(cfg(), lambda m: None)
    pypi["latest"] = "2026.10.1"
    found = ytdlp.candidates(cfg(), lambda m: None)
    assert found[0].version == "2026.9.1"
    found = ytdlp.candidates(cfg(), lambda m: None, refresh=True)
    assert found[0].version == "2026.10.1"
    assert not (cache / "2026.9.1").exists()
    old = time.time() - ytdlp.CHECK_INTERVAL - 1
    os.utime(cache / ".checked", (old, old))
    pypi["latest"] = "2026.11.1"
    assert ytdlp.candidates(cfg(), lambda m: None)[0].version == "2026.11.1"


def test_auto_uses_only_bundled_when_it_is_newest(cache, pypi):
    pypi["latest"] = "2026.8.19"
    assert [y.version for y in ytdlp.candidates(cfg(), lambda m: None)] == [
        "2026.8.19"]
    assert pypi["installs"] == []


def test_auto_survives_pypi_being_unreachable(cache, pypi, monkeypatch):
    ytdlp.candidates(cfg(), lambda m: None)
    def down():
        raise OSError("network unreachable")
    monkeypatch.setattr(ytdlp, "_latest_release", down)
    logged = []
    found = ytdlp.candidates(cfg(), logged.append, refresh=True)
    assert [y.version for y in found] == ["2026.9.1", "2026.8.19"]
    assert any("network unreachable" in m for m in logged)


def test_an_install_that_fails_its_check_is_discarded(cache, pypi, monkeypatch):
    monkeypatch.setattr(ytdlp, "_pip_install",
                        lambda target, v: fake_package(target, "2026.1.1"))
    logged = []
    assert [y.version for y in ytdlp.candidates(cfg(), logged.append)] == [
        "2026.8.19"]
    assert any("failed its check" in m for m in logged)
    assert list(cache.iterdir()) == [cache / ".checked"]


def test_bundled_never_touches_pypi(cache, pypi):
    pypi["latest"] = None
    assert [y.version for y in ytdlp.candidates(cfg("bundled"), print)] == [
        "2026.8.19"]
    assert not cache.exists()


def test_pin_installs_that_release_even_if_older(cache, pypi):
    found = ytdlp.candidates(cfg("2026.07.01"), lambda m: None)
    assert [y.version for y in found] == ["2026.07.01", "2026.8.19"]
    assert pypi["installs"] == ["2026.07.01"]


def test_pin_matching_bundled_installs_nothing(cache, pypi):
    found = ytdlp.candidates(cfg("2026.08.19"), lambda m: None)
    assert [y.version for y in found] == ["2026.8.19"]
    assert pypi["installs"] == []


def test_pin_must_be_a_release_number(cache, pypi):
    with pytest.raises(YtDlpError, match="ytdlp_version"):
        ytdlp.candidates(cfg("latest; rm -rf /"), lambda m: None)
