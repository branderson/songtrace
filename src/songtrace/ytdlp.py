"""The one place songtrace touches yt-dlp.

YouTube breaks old yt-dlp releases faster than songtrace is released, so
besides the copy installed with songtrace (the bundled one), newer stable
releases are fetched from PyPI into a cache dir at runtime and tried first,
with the bundled copy as the fallback. yt-dlp runs in a subprocess and is
driven through its command line, its most stable interface, so a cached
release never mixes with anything songtrace has imported.
"""
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Config
from .pipeline import PipelineError

# The extras are what make YouTube work; see pyproject.toml.
REQUIREMENT = "yt-dlp[default,curl-cffi,deno]"
CHECK_INTERVAL = 24 * 3600
_lock = threading.Lock()


class YtDlpError(PipelineError):
    pass


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    # Installs hold compiled wheels, so they are only good for one Python.
    return Path(base) / "songtrace" / f"yt-dlp-py{sys.version_info[0]}{sys.version_info[1]}"


@dataclass(frozen=True)
class YtDlp:
    version: str
    path: Path | None = None  # None for the bundled copy

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.path:
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, [str(self.path), env.get("PYTHONPATH")]))
            env["PATH"] = os.pathsep.join(
                [str(self.path / "bin"), env.get("PATH", "")])
        return env

    def run(self, args: list[str], log) -> None:
        """Runs yt-dlp with `args`, passing each line it prints to `log`."""
        proc = subprocess.Popen(
            [sys.executable, "-P", "-m", "yt_dlp", "--ignore-config", *args],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors="replace", env=self._env())
        last = error = ""
        for line in proc.stdout:
            last = line.rstrip()
            log(last)
            if last.startswith("ERROR:"):
                error = last
        if proc.wait() == 0:
            return
        if proc.returncode == 2:
            raise YtDlpError(f"yt-dlp {self.version} rejected songtrace's "
                             f"options, so it may be incompatible: {last}")
        raise YtDlpError(error or f"yt-dlp exited with status {proc.returncode}")


def bundled() -> YtDlp:
    return YtDlp(importlib.metadata.version("yt-dlp"))


def _latest_release() -> str:
    r = httpx.get("https://pypi.org/pypi/yt-dlp/json", timeout=15,
                  follow_redirects=True)
    r.raise_for_status()
    return r.json()["info"]["version"]


def _pip_install(target: Path, version: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-input",
         "--disable-pip-version-check", "--target", str(target),
         f"{REQUIREMENT}=={version}"],
        capture_output=True, text=True)
    if proc.returncode:
        raise YtDlpError(f"pip could not install yt-dlp {version}: "
                         f"{proc.stderr.strip()[-500:]}")


def _install(version: str, log) -> YtDlp:
    """Installs `version` into the cache unless it is already there.

    The install is checked to run before it is moved into place, so a
    directory in the cache is always a usable yt-dlp."""
    dest = cache_dir() / version
    if dest.is_dir():
        return YtDlp(version, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"installing yt-dlp {version}")
    tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".tmp-"))
    try:
        _pip_install(tmp, version)
        proc = subprocess.run(
            [sys.executable, "-P", "-m", "yt_dlp", "--version"],
            capture_output=True, text=True, env=YtDlp(version, tmp)._env())
        got = proc.stdout.strip()
        if proc.returncode or _key(got) != _key(version):
            raise YtDlpError(f"yt-dlp {version} failed its check after install: "
                             f"{(got or proc.stderr.strip())[-500:]}")
        try:
            tmp.rename(dest)
        except OSError:
            if not dest.is_dir():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return YtDlp(version, dest)


def _cached() -> list[YtDlp]:
    root = cache_dir()
    if not root.is_dir():
        return []
    found = [YtDlp(p.name, p) for p in root.iterdir()
             if p.is_dir() and re.fullmatch(r"\d+(\.\d+)*", p.name)]
    return sorted(found, key=lambda y: _key(y.version))


def _newest(base: YtDlp, refresh: bool, log) -> YtDlp | None:
    """The newest cached release newer than `base`, checking PyPI for a newer
    one first if that has not been done for a day or `refresh` is set."""
    marker = cache_dir() / ".checked"
    if (refresh or not marker.exists()
            or time.time() - marker.stat().st_mtime > CHECK_INTERVAL):
        # Recorded even on failure, so a broken release is not reinstalled
        # for every job; a failed download refreshes regardless.
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        try:
            latest = _latest_release()
            if _key(latest) > _key(base.version):
                _install(latest, log)
        except Exception as e:
            log(f"could not update yt-dlp: {e}")
    newer = [y for y in _cached() if _key(y.version) > _key(base.version)]
    for old in newer[:-1]:
        shutil.rmtree(old.path, ignore_errors=True)
    return newer[-1] if newer else None


def candidates(cfg: Config, log, refresh: bool = False) -> list[YtDlp]:
    """The yt-dlp copies to try, in order: the newest release (or the one
    `cfg.ytdlp_version` pins), then the bundled copy."""
    base = bundled()
    want = cfg.ytdlp_version
    if want == "bundled":
        return [base]
    with _lock:
        if want == "auto":
            first = _newest(base, refresh, log)
        elif not re.fullmatch(r"\d+(\.\d+)*", want):
            raise YtDlpError(f"ytdlp_version must be \"auto\", \"bundled\" or a "
                             f"release such as \"2026.08.19\", not {want!r}")
        elif _key(want) == _key(base.version):
            first = None
        else:
            try:
                first = _install(want, log)
            except Exception as e:
                log(f"could not install the pinned yt-dlp {want}: {e}")
                first = None
    return [first, base] if first else [base]
