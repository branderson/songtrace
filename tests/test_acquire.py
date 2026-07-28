import json
from pathlib import Path
from songtrace.config import Config
from songtrace.pipeline import Ctx, SongInput
from songtrace.stages import acquire

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

def test_outputs_are_meta_and_audio(tmp_path):
    """_require_source in web.py hardcodes these filenames; pin them here."""
    ctx = make_ctx(tmp_path, youtube_url="https://yt/x")
    outputs = acquire.acquire_stage().outputs(ctx)
    assert [p.name for p in outputs] == ["meta.json", "audio.wav"]

def test_requires_exactly_one_source(tmp_path):
    import pytest
    from songtrace.pipeline import PipelineError
    ctx = make_ctx(tmp_path)
    with pytest.raises(PipelineError):
        acquire.acquire_stage().run(ctx)

def test_classify_ordinary_webm_and_jpg(tmp_path):
    (tmp_path / "source_video.webm").write_bytes(b"v")
    (tmp_path / "source_video.jpg").write_bytes(b"t")
    video, thumb = acquire._classify_downloads(tmp_path)
    assert video.name == "source_video.webm"
    assert thumb.name == "source_video.jpg"

def test_classify_unexpected_thumbnail_extension(tmp_path):
    (tmp_path / "source_video.webm").write_bytes(b"v")
    (tmp_path / "source_video.avif").write_bytes(b"t")
    video, thumb = acquire._classify_downloads(tmp_path)
    assert video.name == "source_video.webm"
    assert thumb.name == "source_video.avif"

def test_classify_no_thumbnail(tmp_path):
    (tmp_path / "source_video.mp4").write_bytes(b"v")
    video, thumb = acquire._classify_downloads(tmp_path)
    assert video.name == "source_video.mp4"
    assert thumb is None

def test_classify_no_video_raises(tmp_path):
    import pytest
    from songtrace.pipeline import PipelineError
    (tmp_path / "source_video.jpg").write_bytes(b"t")
    with pytest.raises(PipelineError):
        acquire._classify_downloads(tmp_path)

def test_classify_picks_largest_video_when_multiple(tmp_path):
    (tmp_path / "source_video.f137.mp4").write_bytes(b"small")
    (tmp_path / "source_video.mp4").write_bytes(b"much larger contents here")
    video, _ = acquire._classify_downloads(tmp_path)
    assert video.name == "source_video.mp4"
