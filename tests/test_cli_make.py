import os

import pytest

import karaoke_maker.cli as cli
from karaoke_maker.config import Config
from karaoke_maker.pipeline import PipelineError, PipelineResult


def setup_cfg(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_config",
                        lambda path=None: Config(whisper_url="u"))

def test_make_success(tmp_path, monkeypatch, capsys):
    setup_cfg(monkeypatch, tmp_path)
    lyrics = tmp_path / "l.txt"
    lyrics.write_text("la la")
    captured = {}
    def fake_run(inp, cfg, log=print, force_from=None, stages=None):
        captured["inp"] = inp
        captured["force_from"] = force_from
        return PipelineResult(output_dir=tmp_path / "out", warnings=["w1"])
    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    rc = cli.main(["make", "--title", "T", "--artist", "A",
                   "--lyrics", str(lyrics), "--youtube", "http://y",
                   "--force-from", "align"])
    assert rc == 0
    assert captured["inp"].lyrics_text == "la la"
    assert captured["force_from"] == "align"
    out = capsys.readouterr().out
    assert "w1" in out and str(tmp_path / "out") in out

def test_make_pipeline_error(tmp_path, monkeypatch, capsys):
    setup_cfg(monkeypatch, tmp_path)
    lyrics = tmp_path / "l.txt"
    lyrics.write_text("x")
    def boom(*a, **kw):
        raise PipelineError("lyrics don't match")
    monkeypatch.setattr(cli, "run_pipeline", boom)
    rc = cli.main(["make", "--title", "T", "--artist", "A",
                   "--lyrics", str(lyrics), "--youtube", "http://y"])
    assert rc == 1
    assert "lyrics don't match" in capsys.readouterr().err

def test_make_requires_one_source(tmp_path, monkeypatch, capsys):
    setup_cfg(monkeypatch, tmp_path)
    lyrics = tmp_path / "l.txt"
    lyrics.write_text("x")
    rc = cli.main(["make", "--title", "T", "--artist", "A",
                   "--lyrics", str(lyrics)])
    assert rc != 0

def test_make_rejects_zero_bpm(tmp_path, monkeypatch, capsys):
    setup_cfg(monkeypatch, tmp_path)
    lyrics = tmp_path / "l.txt"
    lyrics.write_text("x")
    try:
        cli.main(["make", "--title", "T", "--artist", "A",
                  "--lyrics", str(lyrics), "--youtube", "http://y",
                  "--bpm", "0"])
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code != 0
    assert "--bpm" in capsys.readouterr().err

def test_make_rejects_negative_bpm(tmp_path, monkeypatch, capsys):
    setup_cfg(monkeypatch, tmp_path)
    lyrics = tmp_path / "l.txt"
    lyrics.write_text("x")
    try:
        cli.main(["make", "--title", "T", "--artist", "A",
                  "--lyrics", str(lyrics), "--youtube", "http://y",
                  "--bpm", "-120"])
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code != 0
    assert "--bpm" in capsys.readouterr().err

def test_make_reports_missing_lyrics_file(tmp_path, monkeypatch, capsys):
    setup_cfg(monkeypatch, tmp_path)
    def boom(*a, **kw):
        raise AssertionError("run_pipeline must not run with an unreadable lyrics file")
    monkeypatch.setattr(cli, "run_pipeline", boom)
    rc = cli.main(["make", "--title", "T", "--artist", "A",
                   "--lyrics", str(tmp_path / "missing.txt"), "--youtube", "http://y"])
    assert rc != 0
    assert "--lyrics" in capsys.readouterr().err

@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_make_reports_unreadable_lyrics_file(tmp_path, monkeypatch, capsys):
    setup_cfg(monkeypatch, tmp_path)
    lyrics = tmp_path / "l.txt"
    lyrics.write_text("x")
    lyrics.chmod(0o000)
    def boom(*a, **kw):
        raise AssertionError("run_pipeline must not run with an unreadable lyrics file")
    monkeypatch.setattr(cli, "run_pipeline", boom)
    try:
        rc = cli.main(["make", "--title", "T", "--artist", "A",
                       "--lyrics", str(lyrics), "--youtube", "http://y"])
    finally:
        lyrics.chmod(0o644)
    assert rc != 0
    assert "--lyrics" in capsys.readouterr().err

def test_make_reports_undecodable_lyrics_file(tmp_path, monkeypatch, capsys):
    setup_cfg(monkeypatch, tmp_path)
    lyrics = tmp_path / "l.txt"
    lyrics.write_bytes(b"\xff\xfe\x00bad")
    def boom(*a, **kw):
        raise AssertionError("run_pipeline must not run with undecodable lyrics")
    monkeypatch.setattr(cli, "run_pipeline", boom)
    rc = cli.main(["make", "--title", "T", "--artist", "A",
                   "--lyrics", str(lyrics), "--youtube", "http://y"])
    assert rc != 0
    assert "--lyrics" in capsys.readouterr().err
