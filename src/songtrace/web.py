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

from . import rebuild as rebuild_policy
from .config import Config
from .pipeline import (STAGE_ORDER, PipelineError, SongInput, run_pipeline,
                       song_slug)

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
    rebuilt_from: str | None = None
    force_stage: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


def _safe_filename(name: str | None, default: str) -> str:
    safe = Path(name).name if name else ""
    return safe if safe not in ("", ".", "..") else default


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    with dest.open("wb") as f:
        await run_in_threadpool(shutil.copyfileobj, upload.file, f)


async def _store_uploads(cfg: Config, job_id: str, video: UploadFile | None,
                         cover: UploadFile | None) -> tuple[Path | None, Path | None]:
    updir = cfg.work_dir / "uploads" / job_id
    paths = []
    for upload, default in ((video, "video"), (cover, "cover")):
        if not (upload and upload.filename):
            paths.append(None)
            continue
        updir.mkdir(parents=True, exist_ok=True)
        dest = updir / _safe_filename(upload.filename, default)
        await _save_upload(upload, dest)
        paths.append(dest)
    return paths[0], paths[1]


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

    def _prepare_rebuild(job: Job, log) -> str | None:
        """Relocate cached work and report the stage this rebuild restarts from."""
        if job.force_stage:
            log(f"[rebuild] restarting from {job.force_stage} (forced rebuild from scratch)")
            return job.force_stage
        if not job.rebuilt_from:
            return None
        with lock:
            parent = jobs.get(job.rebuilt_from)
        if parent is None:
            return None
        force_from = rebuild_policy.plan(parent.inp, job.inp)
        changed = rebuild_policy.changed_fields(parent.inp, job.inp)
        rebuild_policy.relocate_work(parent.inp, job.inp, cfg)
        log(f"[rebuild] restarting from {force_from or 'the first incomplete stage'}"
            f" ({', '.join(changed) if changed else 'nothing'} changed)")
        return force_from

    def _drop_superseded_output(job: Job, log) -> None:
        if not job.rebuilt_from:
            return
        with lock:
            parent = jobs.get(job.rebuilt_from)
        if parent is None or parent.status != "done":
            return
        old = rebuild_policy.superseded_output(parent.output_dir, job.output_dir,
                                                cfg.output_dir)
        if old is None:
            return
        shutil.rmtree(old, ignore_errors=True)
        log(f"[rebuild] removed superseded {old.name}")

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
                force_from = _prepare_rebuild(job, log)
                result = runner(job.inp, cfg, log=log, force_from=force_from)
                with lock:
                    job.output_dir = str(result.output_dir)
                    job.warnings = list(result.warnings)
                _drop_superseded_output(job, log)
                with lock:
                    job.status = "done"
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

    def _enqueue(job_id: str, inp: SongInput, rebuilt_from: str | None = None,
                 force_stage: str | None = None) -> dict:
        job = Job(id=job_id, title=inp.title, artist=inp.artist, inp=inp,
                  rebuilt_from=rebuilt_from, force_stage=force_stage)
        with lock:
            jobs[job_id] = job
            order.append(job_id)
        todo.put(job_id)
        return {"id": job_id}

    def _finished_job(job_id: str) -> Job:
        with lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(404)
            if job.status in ("queued", "running"):
                raise HTTPException(409, "job is still in progress")
            return job

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
        youtube_url, genre, language = (youtube_url or None, genre or None,
                                        language or None)
        has_video = bool(video and video.filename)
        if bool(youtube_url) == has_video:
            raise HTTPException(400, "provide exactly one of youtube_url / video")
        if bpm is not None and (not math.isfinite(bpm) or bpm <= 0):
            raise HTTPException(400, f"bpm must be a finite value > 0, got {bpm!r}")

        job_id = uuid.uuid4().hex
        video_path, cover_path = await _store_uploads(cfg, job_id, video, cover)

        inp = SongInput(title=title, artist=artist, lyrics_text=lyrics,
                         youtube_url=youtube_url, video_path=video_path,
                         genre=genre, year=year, language=language,
                         cover_path=cover_path, bpm_override=bpm)
        return _enqueue(job_id, inp)

    @app.post("/api/jobs/{job_id}/rerun", status_code=201)
    def rerun(job_id: str):
        parent = _finished_job(job_id)
        return _enqueue(uuid.uuid4().hex, parent.inp, rebuilt_from=job_id)

    @app.post("/api/jobs/{job_id}/rebuild-full", status_code=201)
    def rebuild_full(job_id: str):
        parent = _finished_job(job_id)
        return _enqueue(uuid.uuid4().hex, parent.inp, rebuilt_from=job_id,
                        force_stage=STAGE_ORDER[0])

    def _require_source(parent_inp: SongInput, inp: SongInput,
                        force_from: str | None) -> None:
        """Refuse a rebuild that will rerun acquire against an upload we no longer have."""
        if inp.video_path is None or inp.video_path.exists():
            return
        if force_from != "acquire":
            # acquire is cached under the old slug until the worker renames it
            for slug in {song_slug(parent_inp), song_slug(inp)}:
                work = cfg.work_dir / slug
                if (work / "meta.json").exists() and (work / "audio.wav").exists():
                    return
        raise HTTPException(400, "the original video upload is no longer available; "
                                 "attach the video file again")

    @app.post("/api/jobs/{job_id}/rebuild", status_code=201)
    async def rebuild(job_id: str, title: str = Form(...), artist: str = Form(...),
                      lyrics: str = Form(...), youtube_url: str | None = Form(None),
                      genre: str | None = Form(None), year: int | None = Form(None),
                      language: str | None = Form(None), bpm: float | None = Form(None),
                      video: UploadFile | None = File(None),
                      cover: UploadFile | None = File(None)):
        parent = _finished_job(job_id)
        # a cleared field arrives as "" from some clients; "" and None must not
        # look like different inputs to plan()
        youtube_url, genre, language = (youtube_url or None, genre or None,
                                        language or None)
        has_video = bool(video and video.filename)
        if youtube_url and has_video:
            raise HTTPException(400, "provide exactly one of youtube_url / video")
        if bpm is not None and (not math.isfinite(bpm) or bpm <= 0):
            raise HTTPException(400, f"bpm must be a finite value > 0, got {bpm!r}")

        new_id = uuid.uuid4().hex
        video_path, cover_path = await _store_uploads(cfg, new_id, video, cover)
        if not youtube_url and not has_video:
            youtube_url, video_path = parent.inp.youtube_url, parent.inp.video_path
        if cover_path is None:
            cover_path = parent.inp.cover_path

        inp = SongInput(title=title, artist=artist, lyrics_text=lyrics,
                        youtube_url=youtube_url, video_path=video_path,
                        genre=genre, year=year, language=language,
                        cover_path=cover_path, bpm_override=bpm)
        _require_source(parent.inp, inp, rebuild_policy.plan(parent.inp, inp))
        return _enqueue(new_id, inp, rebuilt_from=job_id)

    @app.get("/api/stages")
    def list_stages():
        return {"stages": list(STAGE_ORDER)}

    @app.get("/api/jobs")
    def list_jobs():
        with lock:
            return [{"id": j.id, "title": j.title, "artist": j.artist,
                      "status": j.status, "stage": j.stage,
                      "created_at": j.created_at, "started_at": j.started_at,
                      "finished_at": j.finished_at, "rebuilt_from": j.rebuilt_from}
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
                    "finished_at": job.finished_at, "rebuilt_from": job.rebuilt_from}

    @app.get("/api/jobs/{job_id}/input")
    def get_input(job_id: str):
        with lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(404)
            inp = job.inp
        # keys are the form's field names so the page can prefill by name;
        # paths are reduced to filenames, which is all the browser can use.
        return {"title": inp.title, "artist": inp.artist, "lyrics": inp.lyrics_text,
                "youtube_url": inp.youtube_url, "genre": inp.genre, "year": inp.year,
                "language": inp.language, "bpm": inp.bpm_override,
                "video_filename": inp.video_path.name if inp.video_path else None,
                "cover_filename": inp.cover_path.name if inp.cover_path else None}

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str):
        job = _finished_job(job_id)
        slug = song_slug(job.inp)
        shutil.rmtree(cfg.work_dir / slug, ignore_errors=True)
        shutil.rmtree(cfg.output_dir / slug, ignore_errors=True)
        with lock:
            jobs.pop(job_id, None)
            if job_id in order:
                order.remove(job_id)

    return app


def serve(cfg: Config) -> int:
    import uvicorn
    uvicorn.run(create_app(cfg), host=cfg.bind_host, port=cfg.port)
    return 0
