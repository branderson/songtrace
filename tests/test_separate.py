import numpy as np
import soundfile as sf
import torch
from karaoke_maker.config import Config
from karaoke_maker.pipeline import Ctx, SongInput
from karaoke_maker.stages import separate


def test_separate_writes_stems(tmp_path, monkeypatch):
    work = tmp_path / "w"
    work.mkdir()
    (work / "audio.wav").write_bytes(b"")
    stems = {
        "vocals": torch.ones(2, 100) * 0.5,
        "drums": torch.ones(2, 100) * 0.1,
        "bass": torch.ones(2, 100) * 0.1,
        "other": torch.ones(2, 100) * 0.1,
    }
    monkeypatch.setattr(separate, "separate_audio", lambda p, m, d: stems)
    ctx = Ctx(inp=SongInput(title="T", artist="A", lyrics_text="x"),
              cfg=Config(whisper_url="u"), work=work, log=lambda m: None)
    separate.separate_stage().run(ctx)
    vocals, sr = sf.read(work / "vocals.wav")
    assert sr == 44100 and vocals.shape == (100, 2)
    instrumental, _ = sf.read(work / "instrumental.wav")
    assert np.allclose(instrumental, 0.3, atol=1e-4)
