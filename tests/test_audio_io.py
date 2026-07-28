import numpy as np
import soundfile as sf

from songtrace.audio_io import load_wav


def test_load_wav_mono(tmp_path):
    path = tmp_path / "mono.wav"
    data = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    sf.write(path, data, 16000)
    waveform, sr = load_wav(path)
    assert sr == 16000
    assert waveform.shape == (1, 1600)
    assert np.allclose(waveform[0].numpy(), data, atol=1e-4)


def test_load_wav_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    left = np.linspace(-0.5, 0.5, 800, dtype=np.float32)
    right = np.linspace(0.5, -0.5, 800, dtype=np.float32)
    sf.write(path, np.stack([left, right], axis=1), 16000)
    waveform, sr = load_wav(path)
    assert sr == 16000
    assert waveform.shape == (2, 800)
    assert np.allclose(waveform[0].numpy(), left, atol=1e-4)
    assert np.allclose(waveform[1].numpy(), right, atol=1e-4)
