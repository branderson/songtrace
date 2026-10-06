import shutil
from pathlib import Path

from .config import Config
from .pipeline import STAGE_ORDER, PipelineError, SongInput, song_slug

# The earliest stage that reads each SongInput field. Editing a field means every
# stage from here onward is stale. Adding a field to SongInput without adding it
# here would let an edit invalidate nothing at all, so tests pin this to the
# dataclass.
FIELD_STAGE = {
    "youtube_url": "acquire",
    "video_path": "acquire",
    "language": "transcribe",
    "lyrics_text": "align",
    "title": "grid",
    "artist": "grid",
    "genre": "grid",
    "year": "grid",
    "bpm_override": "grid",
    "cover_path": "package",
}


def changed_fields(old: SongInput, new: SongInput) -> list[str]:
    return [f for f in FIELD_STAGE if getattr(old, f) != getattr(new, f)]


def plan(old: SongInput, new: SongInput) -> str | None:
    """The stage to restart from, or None when nothing changed.

    None is not "do nothing": run_pipeline with no force_from resumes at the
    first stage whose outputs are missing, which is exactly what a rerun wants.
    """
    changed = changed_fields(old, new)
    if not changed:
        return None
    return min((FIELD_STAGE[f] for f in changed), key=STAGE_ORDER.index)


def earliest(*stages: str | None) -> str | None:
    """The earliest of the stages given, ignoring None ("nothing forced").

    An operator's explicit choice is combined with plan()'s rather than
    replacing it: the planner's stage is the latest that is still correct for
    the edit, so a choice may only ever redo more.
    """
    named = [s for s in stages if s]
    return min(named, key=STAGE_ORDER.index) if named else None


def relocate_work(old: SongInput, new: SongInput, cfg: Config) -> bool:
    """Move the cached work dir when title/artist changed its slug.

    Nothing inside a work dir depends on the slug except song.txt, which `grid`
    rewrites — and FIELD_STAGE sends title/artist edits to `grid`. So a retitle
    costs a rename plus grid and package, not the whole GPU pipeline.
    """
    old_dir = cfg.work_dir / song_slug(old)
    new_dir = cfg.work_dir / song_slug(new)
    if old_dir == new_dir or not old_dir.exists():
        return False
    if new_dir.exists():
        raise PipelineError(
            f"a work folder already exists for {song_slug(new)!r}; "
            "rename or remove it before rebuilding")
    new_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_dir), str(new_dir))
    return True


def superseded_output(old_dir: str | None, new_dir: str,
                      output_root: Path) -> Path | None:
    """The previous build's folder, when the rebuild replaced it under a new slug.

    Returns None unless the path is strictly inside output_root, so a bad job
    record cannot turn the caller's removal into an arbitrary delete.
    """
    if not old_dir:
        return None
    old, new = Path(old_dir).resolve(), Path(new_dir).resolve()
    if old == new or output_root.resolve() not in old.parents:
        return None
    return old
