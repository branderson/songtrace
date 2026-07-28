import json
from dataclasses import asdict

import numpy as np

from ..notes import build_notes
from ..pipeline import Ctx, Stage


def _run(ctx: Ctx) -> None:
    rows = json.loads((ctx.work / "alignment.json").read_text())
    data = np.load(ctx.work / "pitch.npz")
    notes, breaks, warnings = build_notes(
        rows, data["times"], data["f0"], data["conf"], ctx.inp.language)
    (ctx.work / "notes_sec.json").write_text(json.dumps({
        "notes": [asdict(n) for n in notes],
        "breaks": breaks,
        "warnings": warnings,
    }))


def notes_stage() -> Stage:
    return Stage(name="notes",
                 outputs=lambda ctx: [ctx.work / "notes_sec.json"],
                 run=_run)
