import json
from dataclasses import asdict

import numpy as np

from ..lyrics import normalize_lyrics
from ..notes import build_notes
from ..pipeline import Ctx, Stage


def _run(ctx: Ctx) -> None:
    rows = json.loads((ctx.work / "alignment.json").read_text())
    data = np.load(ctx.work / "pitch.npz")
    meta_path = ctx.work / "align_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    dropped_lines = meta.get("dropped_lines", [])
    missing_spans = [tuple(s) for s in meta.get("missing_spans", [])]
    ref_lines = normalize_lyrics(ctx.inp.lyrics_text).lines
    notes, breaks, warnings = build_notes(
        rows, data["times"], data["f0"], data["conf"], ctx.inp.language,
        dropped_lines=dropped_lines, missing_spans=missing_spans,
        ref_lines=ref_lines)
    (ctx.work / "notes_sec.json").write_text(json.dumps({
        "notes": [asdict(n) for n in notes],
        "breaks": breaks,
        "warnings": warnings,
    }))


def notes_stage() -> Stage:
    return Stage(name="notes",
                 outputs=lambda ctx: [ctx.work / "notes_sec.json"],
                 run=_run)
