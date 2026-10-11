import json
import math
import queue
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import edits, grid
from . import rebuild as rebuild_policy
from .config import Config
from .lyrics import normalize_lyrics
from .pipeline import (STAGE_ORDER, PipelineError, SongInput, read_warnings,
                       run_pipeline, song_slug)
from .ultrastar import get_header, parse_ultrastar

STATIC = Path(__file__).parent / "static"
STAGE_LINE = re.compile(r"^\[(\w+)\] (?:running|cached, skipping)")
# the song folder names its own media, exactly as the game reads it
MEDIA_HEADERS = {"audio": "MP3", "video": "VIDEO", "cover": "COVER"}


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
    Never raises, so cleaning up cannot fail a job that otherwise succeeded.
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


def _discover_jobs(cfg: Config) -> list[Job]:
    discovered = []
    if not cfg.work_dir.exists():
        return discovered
    for sidecar in cfg.work_dir.glob("*/input.json"):
        try:
            data = json.loads(sidecar.read_text())
            for key in ("video_path", "cover_path"):
                if data.get(key) is not None:
                    data[key] = Path(data[key])
            inp = SongInput(**data)
            slug = song_slug(inp)
            out = cfg.output_dir / slug
            done = out.exists() and any(out.iterdir())
            mtime = sidecar.stat().st_mtime
            warnings = read_warnings(sidecar.parent) if done else []
        except (OSError, ValueError, TypeError):
            continue
        discovered.append(Job(
            id=uuid.uuid4().hex, title=inp.title, artist=inp.artist, inp=inp,
            status="done" if done else "failed",
            output_dir=str(out) if done else None,
            warnings=warnings,
            error=None if done else "interrupted before completion",
            created_at=mtime, finished_at=mtime))
    discovered.sort(key=lambda j: j.created_at)
    return discovered


def create_app(cfg: Config, runner=run_pipeline) -> FastAPI:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    jobs: dict[str, Job] = {}
    order: list[str] = []
    lock = threading.Lock()
    edit_lock = threading.Lock()
    todo: queue.Queue[str] = queue.Queue()

    for job in _discover_jobs(cfg):
        jobs[job.id] = job
        order.append(job.id)

    def _prepare_rebuild(job: Job, log) -> str | None:
        """Relocate cached work and report the stage this rebuild restarts from."""
        if not job.rebuilt_from:
            return job.force_stage
        with lock:
            parent = jobs.get(job.rebuilt_from)
        if parent is None:
            return job.force_stage
        planned = rebuild_policy.plan(parent.inp, job.inp)
        changed = rebuild_policy.changed_fields(parent.inp, job.inp)
        rebuild_policy.relocate_work(parent.inp, job.inp, cfg)
        # a stage the operator asked for can only widen what plan() decided,
        # which is how "redo the packaging" reaches work nothing invalidated
        force_from = rebuild_policy.earliest(planned, job.force_stage)
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

    def _write_sidecar(job: Job) -> None:
        work = cfg.work_dir / song_slug(job.inp)
        work.mkdir(parents=True, exist_ok=True)
        data = asdict(job.inp)
        for key in ("video_path", "cover_path"):
            if data[key] is not None:
                data[key] = str(data[key])
        (work / "input.json").write_text(json.dumps(data))

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

            status, error = "done", None
            try:
                force_from = _prepare_rebuild(job, log)
                _write_sidecar(job)
                result = runner(job.inp, cfg, log=log, force_from=force_from)
                with lock:
                    job.output_dir = str(result.output_dir)
                    job.warnings = list(result.warnings)
                _drop_superseded_output(job, log)
                # uploads survive a failure so the job stays rerunnable; acquire
                # has already copied the video into the work dir by now.
                _cleanup_uploads(cfg, job_id, job.inp.video_path)
            except BaseException as e:
                status, error = "failed", str(e)
            # last, and all at once: a client that sees the job finished must
            # find everything finishing it does already done
            with lock:
                job.status, job.error, job.finished_at = status, error, time.time()

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
        lyrics = lyrics.replace("\r\n", "\n")
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
                      force_from: str | None = Form(None),
                      video: UploadFile | None = File(None),
                      cover: UploadFile | None = File(None)):
        parent = _finished_job(job_id)
        force_from = force_from or None
        if force_from is not None and force_from not in STAGE_ORDER:
            raise HTTPException(400, f"unknown stage {force_from!r}")
        # a cleared field arrives as "" from some clients; "" and None must not
        # look like different inputs to plan()
        youtube_url, genre, language = (youtube_url or None, genre or None,
                                        language or None)
        # a browser submits a textarea with CRLF; without this every rebuild
        # from the page looks like a lyrics edit and redoes the alignment
        lyrics = lyrics.replace("\r\n", "\n")
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
        _require_source(parent.inp, inp, rebuild_policy.earliest(
            rebuild_policy.plan(parent.inp, inp), force_from))
        return _enqueue(new_id, inp, rebuilt_from=job_id, force_stage=force_from)

    def _current_songs() -> list[Job]:
        """The job that stands for each song now, newest first."""
        with lock:
            # a job that was rebuilt is history: its rebuild is the song now,
            # possibly under another slug
            rebuilt = {j.rebuilt_from for j in jobs.values()}
            latest = {song_slug(j.inp): j for j in (jobs[i] for i in order)
                      if j.id not in rebuilt}
            position = {job_id: n for n, job_id in enumerate(order)}
            return sorted(latest.values(), key=lambda j: position[j.id], reverse=True)

    @app.get("/api/rebuild-all")
    def rebuild_all_candidates():
        return [{"id": j.id, "title": j.title, "artist": j.artist,
                 "status": j.status, "rebuilt_from": j.rebuilt_from,
                 "finished_at": j.finished_at}
                for j in _current_songs()]

    @app.post("/api/rebuild-all", status_code=201)
    def rebuild_all(force_from: str = Form(...), ids: list[str] | None = Form(None)):
        """Rebuild every song, or the ones in `ids`, from `force_from` on, for
        when songtrace itself improved that stage. Each is an ordinary rebuild
        with unchanged inputs."""
        if force_from not in STAGE_ORDER:
            raise HTTPException(400, f"unknown stage {force_from!r}")
        songs = _current_songs()
        queued, skipped = [], []
        if ids is not None:
            current = {j.id for j in songs}
            skipped = [{"id": i, "reason": "no longer the current job for its song"}
                       for i in ids if i not in current]
            songs = [j for j in songs if j.id in set(ids)]
        for job in songs:
            if job.status in ("queued", "running"):
                skipped.append({"id": job.id, "reason": "still in progress"})
                continue
            try:
                _require_source(job.inp, job.inp, force_from)
            except HTTPException as e:
                skipped.append({"id": job.id, "reason": e.detail})
                continue
            queued.append(_enqueue(uuid.uuid4().hex, job.inp, rebuilt_from=job.id,
                                   force_stage=force_from)["id"])
        return {"ids": queued, "skipped": skipped}

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
        with lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(404)
            if job.status in ("queued", "running"):
                raise HTTPException(409, "job is still in progress")
            slug = song_slug(job.inp)
            if any(other.status in ("queued", "running") and song_slug(other.inp) == slug
                   for other_id, other in jobs.items() if other_id != job_id):
                raise HTTPException(409, "another job for this song is in progress")
            jobs.pop(job_id, None)
            if job_id in order:
                order.remove(job_id)
        shutil.rmtree(cfg.work_dir / slug, ignore_errors=True)
        shutil.rmtree(cfg.output_dir / slug, ignore_errors=True)

    def _resolve_media(job: Job, header: str) -> Path | None:
        """The file a song folder's #MP3/#VIDEO/#COVER header points at."""
        slug = song_slug(job.inp)
        out = (cfg.output_dir / slug).resolve()
        txt = out / f"{slug}.txt"
        if not txt.exists():
            return None
        name = get_header(parse_ultrastar(txt.read_text(encoding="utf-8")), header)
        if not name:
            return None
        # a header names a file in the song folder and nothing else
        path = (out / name).resolve()
        if path.parent != out or not path.is_file():
            return None
        return path

    @app.get("/api/jobs/{job_id}/media/{kind}")
    def get_media(job_id: str, kind: str):
        header = MEDIA_HEADERS.get(kind)
        if header is None:
            raise HTTPException(404)
        job = _finished_job(job_id)
        if job.status != "done":
            raise HTTPException(404)
        path = _resolve_media(job, header)
        if path is None:
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/api/jobs/{job_id}/review")
    def get_review(job_id: str):
        job = _finished_job(job_id)
        if job.status != "done":
            raise HTTPException(404)
        work = cfg.work_dir / song_slug(job.inp)
        notes_file = work / edits.FINAL_FILE
        if not notes_file.exists():
            notes_file = work / "notes_sec.json"
        if not notes_file.exists():
            raise HTTPException(404)
        notes_data = json.loads(notes_file.read_text())
        meta_file = work / "align_meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        dropped = set(meta.get("dropped_lines", []))
        missing_spans = [{"start": s, "end": e}
                         for s, e in meta.get("missing_spans", [])]
        align_file = work / "alignment.json"
        align_rows = json.loads(align_file.read_text()) if align_file.exists() else []
        words_by_line: dict[int, list] = {}
        for r in align_rows:
            words_by_line.setdefault(r["line_index"], []).append(
                {"text": r["word"], "start": r["start"], "end": r["end"]})
        ref = normalize_lyrics(job.inp.lyrics_text)
        by_line: dict[int, list] = {}
        for n in notes_data["notes"]:
            by_line.setdefault(n["line_index"], []).append({
                "start": n["start"], "end": n["start"] + n["duration"],
                "pitch": n["pitch"], "low_confidence": n["low_confidence"],
                "text": n["text"], "kind": n.get("kind", ":")})
        edited = set(notes_data.get("edited_lines", []))
        lines = [{"index": i, "text": " ".join(words), "dropped": i in dropped,
                  "edited": i in edited,
                  "notes": by_line.get(i, []), "words": words_by_line.get(i, [])}
                 for i, words in enumerate(ref.lines)]
        stale = [{"text": " ".join(e["words"]), "notes": e["notes"]}
                 for e in notes_data.get("stale", [])]
        media = {kind: _resolve_media(job, header) is not None
                 for kind, header in MEDIA_HEADERS.items()}
        return {"lines": lines, "missing_spans": missing_spans, "media": media,
                "stale": stale, "grid": _chart_grid(work)}

    def _chart_grid(work: Path) -> dict | None:
        """The beat grid the chart was built on, for snapping edits to it."""
        txt = work / "song.txt"
        if not txt.exists():
            return None
        song = parse_ultrastar(txt.read_text(encoding="utf-8"))
        bpm, gap = get_header(song, "BPM"), get_header(song, "GAP")
        if bpm is None or gap is None:
            return None
        return {"bpm": float(bpm.replace(",", ".")), "gap_ms": float(gap.replace(",", "."))}

    def _editable(job_id: str) -> Job:
        job = _finished_job(job_id)
        slug = song_slug(job.inp)
        with lock:
            # a build of this song would overwrite the chart being edited
            if any(j.status in ("queued", "running") and song_slug(j.inp) == slug
                   for j in jobs.values()):
                raise HTTPException(409, "this song is being rebuilt")
        if job.status != "done":
            raise HTTPException(409, "only a finished song can be edited")
        return job

    def _edit(job_id: str, change) -> dict:
        """Apply `change` to the song's edits, then rewrite its charts."""
        job = _editable(job_id)
        work = cfg.work_dir / song_slug(job.inp)
        lyrics = normalize_lyrics(job.inp.lyrics_text).lines
        with edit_lock:
            data = edits.load(work, lyrics)
            try:
                change(data, lyrics)
            except ValueError as e:
                raise HTTPException(400, str(e))
            edits.save(work, data)
            final = edits.finalize(work, lyrics)
            charts = [work / "song.txt"]
            if job.output_dir:
                charts.append(Path(job.output_dir) / f"{song_slug(job.inp)}.txt")
            for txt in charts:
                if txt.exists():
                    grid.rechart(txt, final["notes"], final["breaks"])
        with lock:
            job.warnings = list(final["warnings"])
        return get_review(job_id)

    def _line(lyrics, index: int) -> int:
        if not 0 <= index < len(lyrics):
            raise HTTPException(404, f"no line {index}")
        return index

    @app.put("/api/jobs/{job_id}/lines/{index}")
    def save_line(job_id: str, index: int, body: dict = Body(...)):
        """Save a line as edited: its full note list, in seconds."""
        notes = body.get("notes")
        if not isinstance(notes, list):
            raise HTTPException(400, "body must be {\"notes\": [...]}")
        return _edit(job_id, lambda data, lyrics: edits.set_line(
            data, _line(lyrics, index), notes))

    @app.delete("/api/jobs/{job_id}/lines/{index}")
    def revert_line(job_id: str, index: int):
        """Drop a line's edit, so it follows what the pipeline generates."""
        def change(data, lyrics):
            if not edits.clear_line(data, _line(lyrics, index)):
                raise HTTPException(404, f"line {index} has no edit")
        return _edit(job_id, change)

    @app.delete("/api/jobs/{job_id}/stale/{n}")
    def discard_stale(job_id: str, n: int):
        """Throw away an edit whose line's lyrics have since changed."""
        def change(data, lyrics):
            if not 0 <= n < len(data["stale"]):
                raise HTTPException(404, f"no stale edit {n}")
            data["stale"].pop(n)
        return _edit(job_id, change)

    return app


def serve(cfg: Config) -> int:
    import uvicorn
    uvicorn.run(create_app(cfg), host=cfg.bind_host, port=cfg.port)
    return 0
