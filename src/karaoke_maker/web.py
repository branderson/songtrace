import math
import queue
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .pipeline import STAGE_ORDER, PipelineError, SongInput, run_pipeline

STATIC = Path(__file__).parent / "static"
STAGE_LINE = re.compile(r"^\[(\w+)\] (?:running|cached, skipping)")


@dataclass
class Job:
    id: str
    title: str
    artist: str
    status: str = "queued"
    stage: str | None = None
    log: list[str] = field(default_factory=list)
    output_dir: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    inp: SongInput | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


def _safe_filename(name: str | None, default: str) -> str:
    safe = Path(name).name if name else ""
    return safe if safe not in ("", ".", "..") else default


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    with dest.open("wb") as f:
        await run_in_threadpool(shutil.copyfileobj, upload.file, f)


def _cleanup_uploads(cfg: Config, job_id: str, video_path: Path | None) -> None:
    """Drop the video upload a finished job no longer needs.

    Covers are left behind: they are small, and a rebuild inherits cover_path, so
    removing one would silently strip #COVER from the rebuilt song. A rebuild can
    also inherit its parent's upload, so the directory is not always uploads/<job_id>.
    Never raises — the caller has already marked the job done.
    """
    uploads = (cfg.work_dir / "uploads").resolve()
    dirs = {uploads / job_id}
    if video_path is not None:
        target = Path(video_path)
        parent = target.resolve().parent
        if parent.parent == uploads:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            dirs.add(parent)
    for d in dirs:
        try:
            d.rmdir()
        except OSError:
            pass


def create_app(cfg: Config, runner=run_pipeline) -> FastAPI:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    jobs: dict[str, Job] = {}
    order: list[str] = []
    lock = threading.Lock()
    todo: queue.Queue[str] = queue.Queue()

    def worker():
        while True:
            job_id = todo.get()
            with lock:
                job = jobs[job_id]
                job.status = "running"
                job.started_at = time.time()

            def log(msg: str, job=job):
                with lock:
                    job.log.append(msg)
                    if m := STAGE_LINE.match(msg):
                        job.stage = m[1]

            try:
                result = runner(job.inp, cfg, log=log)
                with lock:
                    job.status = "done"
                    job.output_dir = str(result.output_dir)
                    job.warnings = list(result.warnings)
                # uploads survive a failure so the job stays rerunnable; acquire
                # has already copied the video into the work dir by now.
                _cleanup_uploads(cfg, job_id, job.inp.video_path)
            except BaseException as e:
                with lock:
                    job.status = "failed"
                    job.error = str(e)
            finally:
                with lock:
                    job.finished_at = time.time()

    threading.Thread(target=worker, daemon=True).start()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.post("/api/jobs", status_code=201)
    async def submit(title: str = Form(...), artist: str = Form(...),
                      lyrics: str = Form(...), youtube_url: str | None = Form(None),
                      genre: str | None = Form(None), year: int | None = Form(None),
                      language: str | None = Form(None), bpm: float | None = Form(None),
                      video: UploadFile | None = File(None),
                      cover: UploadFile | None = File(None)):
        has_video = bool(video and video.filename)
        if bool(youtube_url) == has_video:
            raise HTTPException(400, "provide exactly one of youtube_url / video")
        if bpm is not None and (not math.isfinite(bpm) or bpm <= 0):
            raise HTTPException(400, f"bpm must be a finite value > 0, got {bpm!r}")

        job_id = uuid.uuid4().hex
        updir = cfg.work_dir / "uploads" / job_id
        video_path = cover_path = None
        if has_video:
            updir.mkdir(parents=True, exist_ok=True)
            video_path = updir / _safe_filename(video.filename, "video")
            await _save_upload(video, video_path)
        if cover and cover.filename:
            updir.mkdir(parents=True, exist_ok=True)
            cover_path = updir / _safe_filename(cover.filename, "cover")
            await _save_upload(cover, cover_path)

        inp = SongInput(title=title, artist=artist, lyrics_text=lyrics,
                         youtube_url=youtube_url, video_path=video_path,
                         genre=genre, year=year, language=language,
                         cover_path=cover_path, bpm_override=bpm)
        job = Job(id=job_id, title=title, artist=artist, inp=inp)
        with lock:
            jobs[job_id] = job
            order.append(job_id)
        todo.put(job_id)
        return {"id": job_id}

    @app.get("/api/stages")
    def list_stages():
        return {"stages": list(STAGE_ORDER)}

    @app.get("/api/jobs")
    def list_jobs():
        with lock:
            return [{"id": j.id, "title": j.title, "artist": j.artist,
                      "status": j.status, "stage": j.stage,
                      "created_at": j.created_at, "started_at": j.started_at,
                      "finished_at": j.finished_at}
                     for j in (jobs[i] for i in reversed(order))]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        with lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(404)
            return {"id": job.id, "title": job.title, "artist": job.artist,
                    "status": job.status, "stage": job.stage,
                    "log": job.log[-100:], "output_dir": job.output_dir,
                    "warnings": list(job.warnings), "error": job.error,
                    "created_at": job.created_at, "started_at": job.started_at,
                    "finished_at": job.finished_at}

    return app


def serve(cfg: Config) -> int:
    import uvicorn
    uvicorn.run(create_app(cfg), host=cfg.bind_host, port=cfg.port)
    return 0
