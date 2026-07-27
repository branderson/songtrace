from pathlib import Path

import numpy as np

from ..pipeline import Ctx, Stage


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
