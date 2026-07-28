import numpy as np
from songtrace.config import Config
from songtrace.pipeline import Ctx, SongInput
from songtrace.stages import pitch


def test_pitch_stage_writes_npz(tmp_path, monkeypatch):
    work = tmp_path / "w"
    work.mkdir()
    times = np.arange(0, 1.0, 0.01)
    f0 = np.full_like(times, 220.0)
    conf = np.full_like(times, 0.9)
    monkeypatch.setattr(pitch, "track_pitch", lambda p, m, d: (times, f0, conf))
    ctx = Ctx(inp=SongInput(title="T", artist="A", lyrics_text="x"),
              cfg=Config(whisper_url="u"), work=work, log=lambda m: None)
    pitch.pitch_stage().run(ctx)
    data = np.load(work / "pitch.npz")
    assert np.allclose(data["f0"], 220.0)
    assert data["times"].shape == data["conf"].shape
