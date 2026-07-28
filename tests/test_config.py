import pytest
from pathlib import Path
from songtrace.config import Config, load_config


def test_load_minimal(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('whisper_url = "http://w:9000/v1"\n')
    cfg = load_config(cfg_file)
    assert cfg.whisper_url == "http://w:9000/v1"
    assert cfg.demucs_model == "htdemucs_ft"


def test_paths_and_overrides(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        'whisper_url = "http://w/v1"\noutput_dir = "/srv/songs"\nport = 9999\n'
    )
    cfg = load_config(cfg_file)
    assert cfg.output_dir == Path("/srv/songs")
    assert cfg.port == 9999


def test_env_var(tmp_path, monkeypatch):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('whisper_url = "http://env/v1"\n')
    monkeypatch.setenv("SONGTRACE_CONFIG", str(cfg_file))
    assert load_config().whisper_url == "http://env/v1"


def test_unknown_key_rejected(tmp_path):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('whisper_url = "x"\nwhipser_model = "typo"\n')
    with pytest.raises(ValueError, match="whipser_model"):
        load_config(cfg_file)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")
