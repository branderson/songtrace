from pathlib import Path

import soundfile as sf

from ..pipeline import Ctx, Stage


def separate_audio(audio_path: Path, model: str, device: str):
    from demucs.api import Separator
    separator = Separator(model=model, device=device)
    _, stems = separator.separate_audio_file(audio_path)
    return stems


def _run(ctx: Ctx) -> None:
    stems = separate_audio(ctx.work / "audio.wav",
                           ctx.cfg.demucs_model, ctx.cfg.device)
    vocals = stems["vocals"]
    others = [t for name, t in stems.items() if name != "vocals"]
    instrumental = sum(others[1:], start=others[0])
    sf.write(ctx.work / "vocals.wav", vocals.cpu().numpy().T, 44100)
    sf.write(ctx.work / "instrumental.wav", instrumental.cpu().numpy().T, 44100)


def separate_stage() -> Stage:
    return Stage(
        name="separate",
        outputs=lambda ctx: [ctx.work / "vocals.wav", ctx.work / "instrumental.wav"],
        run=_run,
    )
