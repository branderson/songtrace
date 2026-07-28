"""Regenerate docs/images/webui.png against the current web UI.

Seeds the real app (via create_app's fake-runner seam, same as tests/test_web.py)
with a fixed set of jobs covering every status, then screenshots it.

One-time setup: pip install -e '.[dev]' && playwright install chromium
Run: python scripts/screenshot_webui.py
"""
import tempfile
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from playwright.sync_api import sync_playwright

from songtrace.config import Config
from songtrace.pipeline import PipelineError, PipelineResult
from songtrace.web import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_IMAGE = REPO_ROOT / "docs" / "images" / "webui.png"
STAGES = ["acquire", "separate", "transcribe", "align", "pitch", "notes", "grid", "package"]

# order matters: the app has one worker thread, so each job runs in submission
# order. "running" must be submitted after the done/failed ones complete and
# before "queued", so it's the one still in progress when "queued" arrives.
SONGS = [
    ("Every Single Night", "Fiona Apple", "done",
     {"warnings": ["1 low-confidence note near 2:41–2:44"]}),
    ("This Year", "The Mountain Goats", "done", {"warnings": []}),
    ("Pink Moon", "Nick Drake", "done",
     {"warnings": ["2 low-confidence notes near 0:38–0:41",
                    "1 low-confidence note near 1:22–1:24"]}),
    ("Such Great Heights", "The Postal Service", "failed",
     {"error": "yt-dlp: unable to download video data: HTTP Error 403: Forbidden"}),
    ("Time Escaping", "Big Thief", "running", {}),
    ("Vampire in the Corner", "Magdalena Bay", "queued", {}),
]
SELECTED = "Pink Moon"

running_event = threading.Event()


def make_runner():
    specs = {title: (kind, extra) for title, _, kind, extra in SONGS}

    def runner(inp, cfg, log=print, force_from=None, stages=None):
        kind, extra = specs[inp.title]
        if kind == "done":
            for s in STAGES:
                log(f"[{s}] running")
            out = cfg.output_dir / f"{inp.artist} - {inp.title}"
            out.mkdir(parents=True, exist_ok=True)
            return PipelineResult(output_dir=out, warnings=extra["warnings"])
        if kind == "failed":
            log("[acquire] running")
            raise PipelineError(extra["error"])
        for s in ["acquire", "separate", "transcribe", "align"]:
            log(f"[{s}] running")
        running_event.wait()
        return PipelineResult(output_dir=cfg.output_dir / "x", warnings=[])

    return runner


def start_server(tmp_path: Path) -> str:
    cfg = Config(whisper_url="http://localhost:9000/v1", output_dir=tmp_path / "out",
                 work_dir=tmp_path / "work", bind_host="127.0.0.1", port=8811)
    app = create_app(cfg, runner=make_runner())
    server = uvicorn.Server(uvicorn.Config(app, host=cfg.bind_host, port=cfg.port,
                                            log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return f"http://{cfg.bind_host}:{cfg.port}"


def seed_jobs(base_url: str) -> None:
    lyrics = "[verse]\n...\n[chorus]\n...\n"
    with httpx.Client(timeout=10) as client:
        for title, artist, kind, _ in SONGS:
            resp = client.post(f"{base_url}/api/jobs", data={
                "title": title, "artist": artist, "lyrics": lyrics,
                "youtube_url": "https://youtu.be/00000000000"})
            resp.raise_for_status()
            job_id = resp.json()["id"]
            if kind == "queued":
                continue
            for _ in range(200):
                job = client.get(f"{base_url}/api/jobs/{job_id}").json()
                if kind == "running" and job["stage"] == "align":
                    break
                if job["status"] in ("done", "failed"):
                    break
                time.sleep(0.05)


def capture(base_url: str) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(base_url, wait_until="networkidle")
        page.get_by_text(SELECTED, exact=True).click()
        page.wait_for_timeout(300)
        OUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(OUT_IMAGE))
        browser.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_url = start_server(Path(tmp))
        time.sleep(1)
        seed_jobs(base_url)
        capture(base_url)
    print(f"wrote {OUT_IMAGE}")


if __name__ == "__main__":
    main()
