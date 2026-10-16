"""Regenerate docs/images/webui.png against the current web UI.

Seeds the real app (via create_app's fake-runner seam, same as tests/test_web.py)
with a fixed set of jobs covering every status, then screenshots it.

One-time setup: pip install -e '.[dev]' && playwright install chromium
Run: python scripts/screenshot_webui.py
"""
import json
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from playwright.sync_api import sync_playwright

from songtrace.config import Config
from songtrace.grid import to_ultrastar
from songtrace.pipeline import PipelineError, PipelineResult, song_slug
from songtrace.ultrastar import serialize_ultrastar
from songtrace.web import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_IMAGE = REPO_ROOT / "docs" / "images" / "webui.png"
STAGES = ["acquire", "separate", "transcribe", "align", "pitch", "notes", "grid", "package"]
DEFAULT_LYRICS = "[verse]\n...\n[chorus]\n...\n"

# public domain (16th century English folk song, author unknown)
GREENSLEEVES_LYRICS = """\
Alas, my love, you do me wrong
To cast me off discourteously
For I have loved you well and long
Delighting in your company

Greensleeves was all my joy
Greensleeves was my delight
Greensleeves was my heart of gold
And who but my lady Greensleeves

I have been ready at your hand
To grant whatever you would crave
I have both waged life and land
Your love and good will for to have

Greensleeves was all my joy
Greensleeves was my delight
Greensleeves was my heart of gold
And who but my lady Greensleeves
"""

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
    ("Landslide", "Fleetwood Mac", "done",
     {"warnings": ["1 low-confidence note near 0:52–0:55"]}),
    ("Greensleeves", "Traditional", "done",
     {"lyrics": GREENSLEEVES_LYRICS,
      "warnings": ["1 low-confidence note near 0:45–0:47",
                   "1 line dropped (not sung) near \"for i have loved\"",
                   "missing lyrics near 2:10–2:18"],
      "review": {
          "lines": [
              # Pitches are the real Greensleeves melody (E natural minor, as
              # arranged by Gilbert DeBenedetti, "Level 4 - Flowing"), converted
              # to UltraStar semitones (0 = C4 = MIDI 60): E5=16 F#5=18 G5=19
              # A5=21 B4=11 D5=14.
              {"index": 0, "text": "Alas, my love, you do me wrong",
               "dropped": False,
               "notes": [{"start": 10.0, "end": 10.4, "pitch": 16, "low_confidence": False},
                         {"start": 10.4, "end": 10.6, "pitch": 18, "low_confidence": False},
                         {"start": 10.6, "end": 11.2, "pitch": 19, "low_confidence": False},
                         {"start": 11.2, "end": 11.5, "pitch": 21, "low_confidence": False},
                         {"start": 11.5, "end": 11.7, "pitch": 19, "low_confidence": False},
                         {"start": 11.7, "end": 11.9, "pitch": 18, "low_confidence": False},
                         {"start": 11.9, "end": 12.9, "pitch": 14, "low_confidence": False}],
               "words": [{"text": "Alas,", "start": 10.0, "end": 10.4},
                         {"text": "my", "start": 10.4, "end": 10.6},
                         {"text": "love,", "start": 10.6, "end": 11.2},
                         {"text": "you", "start": 11.2, "end": 11.5},
                         {"text": "do", "start": 11.5, "end": 11.7},
                         {"text": "me", "start": 11.7, "end": 11.9},
                         {"text": "wrong", "start": 11.9, "end": 12.9}]},
              {"index": 1, "text": "To cast me off discourteously",
               "dropped": False,
               "notes": [{"start": 44.0, "end": 44.3, "pitch": 11, "low_confidence": False},
                         {"start": 44.3, "end": 44.6, "pitch": 16, "low_confidence": False},
                         {"start": 44.6, "end": 44.8, "pitch": 18, "low_confidence": False},
                         {"start": 44.8, "end": 45.0, "pitch": 19, "low_confidence": False},
                         {"start": 45.0, "end": 45.4, "pitch": 18, "low_confidence": True}],
               "words": [{"text": "To", "start": 44.0, "end": 44.3},
                         {"text": "cast", "start": 44.3, "end": 44.6},
                         {"text": "me", "start": 44.6, "end": 44.8},
                         {"text": "off", "start": 44.8, "end": 45.0},
                         {"text": "discourteously", "start": 45.0, "end": 45.4}]},
              {"index": 2, "text": "For I have loved you well and long",
               "dropped": True, "notes": [], "words": []},
          ],
          "missing_spans": [{"start": 130.0, "end": 138.0}],
      }}),
    ("Such Great Heights", "The Postal Service", "failed",
     {"error": "yt-dlp: unable to download video data: HTTP Error 403: Forbidden"}),
    ("Time Escaping", "Big Thief", "running", {}),
    ("Vampire in the Corner", "Magdalena Bay", "queued", {}),
]
SELECTED = "Greensleeves"
REVIEW_HEADING = "REVIEW"

running_event = threading.Event()


def make_runner():
    specs = {title: (kind, extra) for title, _, kind, extra in SONGS}

    def runner(inp, cfg, log=print, force_from=None, stages=None):
        kind, extra = specs[inp.title]
        if kind == "done":
            for s in STAGES:
                log(f"[{s}] running")
            if "review" in extra:
                work = cfg.work_dir / song_slug(inp)
                work.mkdir(parents=True, exist_ok=True)
                # one note per word in the seeded lines, so each sings its word
                notes = [
                    {**n, "duration": n["end"] - n["start"],
                     "text": (" " if k else "") + w["text"], "line_index": line["index"]}
                    for line in extra["review"]["lines"]
                    for k, (n, w) in enumerate(zip(line["notes"], line["words"]))
                ]
                sung = [line for line in extra["review"]["lines"] if line["notes"]]
                breaks = [(a["notes"][-1]["end"] + b["notes"][0]["start"]) / 2
                          for a, b in zip(sung, sung[1:])]
                (work / "notes_sec.json").write_text(json.dumps({
                    "notes": notes, "breaks": breaks, "warnings": extra["warnings"]}))
                # a chart for the beat grid, and a sung pitch with a little
                # vibrato for the curve over the notes
                (work / "song.txt").write_text(serialize_ultrastar(to_ultrastar(
                    notes, breaks, 240.0, [("TITLE", inp.title)])), encoding="utf-8")
                times = np.arange(0, 60, 0.01)
                f0, conf = np.zeros_like(times), np.zeros_like(times)
                for n in notes:
                    m = (times >= n["start"] + 0.02) & (times < n["start"] + n["duration"] - 0.02)
                    semis = n["pitch"] + 0.2 * np.sin(2 * np.pi * 5.5 * times[m])
                    f0[m], conf[m] = 440 * 2 ** ((semis - 9) / 12), 0.9
                np.savez(work / "pitch.npz", times=times, f0=f0, conf=conf)
                (work / "align_meta.json").write_text(json.dumps({
                    "dropped_lines": [line["index"] for line in extra["review"]["lines"]
                                      if line["dropped"]],
                    "missing_spans": [[s["start"], s["end"]]
                                      for s in extra["review"]["missing_spans"]],
                }))
                (work / "alignment.json").write_text(json.dumps([
                    {"word": w["text"], "start": w["start"], "end": w["end"],
                     "score": 0.9, "line_index": line["index"]}
                    for line in extra["review"]["lines"] for w in line.get("words", [])
                ]))
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


def start_server(tmp_path: Path) -> tuple[str, Config]:
    cfg = Config(whisper_url="http://localhost:9000/v1", output_dir=tmp_path / "out",
                 work_dir=tmp_path / "work", bind_host="127.0.0.1", port=8811)
    app = create_app(cfg, runner=make_runner())
    server = uvicorn.Server(uvicorn.Config(app, host=cfg.bind_host, port=cfg.port,
                                            log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return f"http://{cfg.bind_host}:{cfg.port}", cfg


def install_media(cfg: Config) -> bool:
    """Give the selected song a playable folder, so the shot shows the player.

    A flat fill rather than a test pattern: the point is the transport and the
    lyric tracking underneath it, not the picture. Skipped without ffmpeg.
    """
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found; screenshotting without the player")
        return False
    artist, title = next((a, t) for t, a, _, _ in SONGS if t == SELECTED)
    slug = f"{artist} - {title}"
    out = cfg.output_dir / slug
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=c=0x1A1430:s=640x360:d=60", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-an",
         str(out / f"{slug}.mp4")], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "anullsrc=r=44100:cl=stereo:d=60", "-b:a", "64k",
         str(out / f"{slug}.mp3")], check=True)
    (out / f"{slug}.txt").write_text(
        f"#TITLE:{title}\n#ARTIST:{artist}\n#MP3:{slug}.mp3\n#VIDEO:{slug}.mp4\n"
        f"#BPM:240\n#GAP:10000\n: 0 4 16 Alas,\nE\n", encoding="utf-8")
    return True


def seed_jobs(base_url: str) -> None:
    with httpx.Client(timeout=10) as client:
        for title, artist, kind, extra in SONGS:
            resp = client.post(f"{base_url}/api/jobs", data={
                "title": title, "artist": artist,
                "lyrics": extra.get("lyrics", DEFAULT_LYRICS),
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


def capture(base_url: str, with_player: bool) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1000})
        page.goto(base_url, wait_until="networkidle")
        page.get_by_text(SELECTED, exact=True).click()
        page.wait_for_timeout(500)
        if with_player:
            # paused mid-line, so the shot carries the playhead and the sung /
            # current word colouring rather than an untouched transport
            page.evaluate("document.getElementById('player-media').currentTime = 11.6")
            page.wait_for_timeout(400)
        page.wait_for_selector(".sheet-curve polyline")
        # scrolled to the review, the panel in use, with the note sheet whole
        page.evaluate("""() => {
          const main = document.getElementById('main');
          main.scrollTo(0, document.getElementById('review').offsetTop - 12);
        }""")
        page.wait_for_timeout(400)
        OUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(OUT_IMAGE))
        browser.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_url, cfg = start_server(Path(tmp))
        time.sleep(1)
        seed_jobs(base_url)
        with_player = install_media(cfg)
        capture(base_url, with_player)
    print(f"wrote {OUT_IMAGE}")


if __name__ == "__main__":
    main()
