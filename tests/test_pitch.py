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


def test_mute_silence_drops_bleed_but_keeps_quiet_singing():
    # 60 frames of singing around -30 dB, one quiet sung frame at -50 dB, and
    # 40 frames of bleed at -60 dB: only the bleed is more than 25 dB below
    # the song's loud level.
    loudness = np.array([-30.0] * 60 + [-50.0] + [-60.0] * 40)
    conf = np.full_like(loudness, 0.8)
    pitch.mute_silence(conf, loudness)
    assert (conf[:61] == 0.8).all()
    assert (conf[61:] == 0.0).all()


def test_mute_silence_is_relative_to_the_songs_level():
    loudness = np.array([-45.0] * 60 + [-75.0] * 40)
    conf = np.full_like(loudness, 0.8)
    pitch.mute_silence(conf, loudness)
    assert (conf[:60] == 0.8).all() and (conf[60:] == 0.0).all()
