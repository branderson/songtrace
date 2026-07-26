import json
from pathlib import Path
from karaoke_maker.config import Config
from karaoke_maker.pipeline import Ctx, SongInput
from karaoke_maker.stages import acquire

def make_ctx(tmp_path, **kw):
    inp = SongInput(title="T", artist="A", lyrics_text="x", **kw)
    cfg = Config(whisper_url="http://x/v1", work_dir=tmp_path)
    work = tmp_path / "A - T"
    work.mkdir()
    return Ctx(inp=inp, cfg=cfg, work=work, log=lambda m: None)

def test_local_video(tmp_path, monkeypatch):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    calls = []
    monkeypatch.setattr(acquire, "ffmpeg", lambda *a: calls.append(a))
    ctx = make_ctx(tmp_path, video_path=src)
    acquire.acquire_stage().run(ctx)
    meta = json.loads((ctx.work / "meta.json").read_text())
    assert meta["video_file"] == "source_video.mp4"
    assert (ctx.work / "source_video.mp4").read_bytes() == b"fake"
    assert any("audio.wav" in str(a[-1]) for a in calls)

def test_youtube(tmp_path, monkeypatch):
    def fake_dl(url, dest):
        video = dest / "source_video.webm"
        video.write_bytes(b"v")
        thumb = dest / "source_video.jpg"
        thumb.write_bytes(b"t")
        return video, thumb
    monkeypatch.setattr(acquire, "download_youtube", fake_dl)
    monkeypatch.setattr(acquire, "ffmpeg", lambda *a: None)
    ctx = make_ctx(tmp_path, youtube_url="https://yt/x")
    acquire.acquire_stage().run(ctx)
    meta = json.loads((ctx.work / "meta.json").read_text())
    assert meta["video_file"] == "source_video.webm"
    assert meta["thumbnail"] == "source_video.jpg"

def test_requires_exactly_one_source(tmp_path):
    import pytest
    from karaoke_maker.pipeline import PipelineError
    ctx = make_ctx(tmp_path)
    with pytest.raises(PipelineError):
        acquire.acquire_stage().run(ctx)
