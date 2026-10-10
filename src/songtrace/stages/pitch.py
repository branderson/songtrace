import sys
from pathlib import Path

import numpy as np

from ..models import on_release
from ..pipeline import Ctx, Stage

# Separated vocal stems keep faint bleed that crepe still finds a pitch in, so
# frames this far below the song's loud vocal level count as silence. Relative
# rather than absolute, so it holds however loud the song was mastered.
SILENCE_BELOW_LOUD_DB = 25.0


@on_release
def unload_model() -> None:
    """torchcrepe caches its model as attributes on its own infer function."""
    torchcrepe = sys.modules.get("torchcrepe")
    if torchcrepe is None:
        return
    for attr in ("model", "capacity"):
        if hasattr(torchcrepe.infer, attr):
            delattr(torchcrepe.infer, attr)


def mute_silence(conf: np.ndarray, loudness: np.ndarray) -> None:
    """Zero the confidence of frames quiet enough to be bleed, in place."""
    conf[loudness < np.quantile(loudness, 0.9) - SILENCE_BELOW_LOUD_DB] = 0.0


def track_pitch(vocals_path: Path, model: str, device: str):
    import torchaudio
    import torchcrepe

    from ..audio_io import load_wav

    waveform, sr = load_wav(vocals_path)
    waveform = waveform.mean(0, keepdim=True)
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
    f0, periodicity = torchcrepe.predict(
        waveform, 16000, hop_length=160, model=model,
        fmin=50.0, fmax=1100.0, device=device,
        return_periodicity=True, batch_size=512)
    f0 = f0[0].cpu().numpy()
    conf = periodicity[0].cpu().numpy()
    mute_silence(conf, torchcrepe.loudness.a_weighted(waveform, 16000, 160)[0].cpu().numpy())
    times = np.arange(len(f0)) * 0.01
    return times, f0, conf


def _run(ctx: Ctx) -> None:
    times, f0, conf = track_pitch(ctx.work / "vocals.wav",
                                  ctx.cfg.crepe_model, ctx.cfg.device)
    np.savez(ctx.work / "pitch.npz", times=times, f0=f0, conf=conf)


def pitch_stage() -> Stage:
    return Stage(name="pitch",
                 outputs=lambda ctx: [ctx.work / "pitch.npz"],
                 run=_run)
