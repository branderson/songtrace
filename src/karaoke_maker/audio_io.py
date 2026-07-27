from pathlib import Path

import soundfile as sf
import torch


def load_wav(path: Path) -> tuple[torch.Tensor, int]:
    """Load a WAV file as a (channels, samples) float32 tensor.

    Reads via soundfile rather than torchaudio.load, whose default backend
    now requires the separate torchcodec package.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T).contiguous(), sr
