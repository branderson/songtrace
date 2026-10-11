import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import models
from .config import Config

class PipelineError(Exception):
    pass

@dataclass
class SongInput:
    title: str
    artist: str
    lyrics_text: str
    youtube_url: str | None = None
    video_path: Path | None = None
    genre: str | None = None
    year: int | None = None
    language: str | None = None
    cover_path: Path | None = None      # local file or URL already fetched by front-end
    bpm_override: float | None = None

@dataclass
class Stage:
    name: str
    outputs: Callable[["Ctx"], list[Path]]
    run: Callable[["Ctx"], None]

@dataclass
class Ctx:
    inp: SongInput
    cfg: Config
    work: Path                          # per-song work dir (created)
    log: Callable[[str], None]

@dataclass
class PipelineResult:
    output_dir: Path
    warnings: list[str] = field(default_factory=list)

STAGE_ORDER = ["acquire", "separate", "transcribe", "align", "pitch", "notes", "grid", "package"]

def song_slug(inp: SongInput) -> str:
    name = f"{inp.artist} - {inp.title}"
    return name.replace("/", "").replace("\0", "")

def read_warnings(work: Path) -> list[str]:
    # grid's copy accounts for the user's edits; notes' is all a song built
    # before edits existed has
    notes_file = work / "notes_final.json"
    if not notes_file.exists():
        notes_file = work / "notes_sec.json"
    if not notes_file.exists():
        return []
    return json.loads(notes_file.read_text()).get("warnings", [])

def _real_stages() -> list[Stage]:
    from . import stages
    return stages.all_stages()

def run_pipeline(inp: SongInput, cfg: Config, log=print,
                 force_from: str | None = None,
                 stages: list[Stage] | None = None) -> PipelineResult:
    if stages is None:
        stages = _real_stages()
    names = [s.name for s in stages]
    if force_from is not None and force_from not in names:
        raise PipelineError(f"unknown stage {force_from!r}, expected one of {names}")
    work = cfg.work_dir / song_slug(inp)
    work.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(inp=inp, cfg=cfg, work=work, log=log)
    forcing = False
    for stage in stages:
        if stage.name == force_from:
            forcing = True
        if not forcing and all(p.exists() for p in stage.outputs(ctx)):
            log(f"[{stage.name}] cached, skipping")
            continue
        log(f"[{stage.name}] running")
        # per stage rather than per job: each model is used by one stage, so
        # holding it past that only keeps VRAM from whatever else wants it
        try:
            stage.run(ctx)
        finally:
            # logged only when it actually gave something back, so the line is
            # evidence rather than noise on a box where this matters
            if freed := models.release():
                log(f"[{stage.name}] released {freed / 2**20:.0f} MiB of GPU memory")
    return PipelineResult(output_dir=cfg.output_dir / song_slug(inp),
                          warnings=read_warnings(work))
