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
    def fake_dl(url, dest, cfg, log):
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

def test_classify_ignores_partials_left_by_a_failed_attempt(tmp_path):
    (tmp_path / "source_video.webm").write_bytes(b"v")
    (tmp_path / "source_video.f137.mp4.part").write_bytes(b"a much bigger partial")
    (tmp_path / "source_video.f251.webm.ytdl").write_bytes(b"resume state")
    video, _ = acquire._classify_downloads(tmp_path)
    assert video.name == "source_video.webm"

def test_ydl_args_default_client_adds_no_player_client(tmp_path):
    cfg = Config(whisper_url="http://x/v1")
    args = acquire._ydl_args(cfg, tmp_path, "default", "https://yt/x")
    assert "--extractor-args" not in args
    assert "--retries" in args and "--fragment-retries" in args
    assert args[-2:] == ["--", "https://yt/x"]

def test_ydl_args_named_client_pot_provider_and_cookies(tmp_path):
    cfg = Config(whisper_url="http://x/v1",
                 ytdlp_pot_provider_url="http://pot:4416",
                 ytdlp_cookies=tmp_path / "cookies.txt",
                 ytdlp_impersonate="chrome")
    args = acquire._ydl_args(cfg, tmp_path, "tv", "https://yt/x")
    joined = " ".join(args)
    assert "--extractor-args youtube:player_client=tv" in joined
    assert ("--extractor-args youtubepot-bgutilhttp:base_url=http://pot:4416"
            in joined)
    assert f"--cookies {tmp_path / 'cookies.txt'}" in joined
    assert "--impersonate chrome" in joined

def test_ydl_args_parse_with_real_yt_dlp(tmp_path):
    """Every option songtrace passes must exist in the bundled yt-dlp."""
    import pytest
    pytest.importorskip("yt_dlp")
    from yt_dlp import parse_options
    cfg = Config(whisper_url="http://x/v1", ytdlp_pot_provider_url="http://p:1",
                 ytdlp_cookies=tmp_path / "c.txt")
    parse_options(acquire._ydl_args(cfg, tmp_path, "tv", "https://yt/x"))

class FakeYtDlp:
    def __init__(self, version, behaviour, seen):
        self.version, self._behaviour, self._seen = version, behaviour, seen
    def run(self, args, log):
        client = next((a for a in args if a.startswith("youtube:")), None)
        self._seen.append((self.version, client))
        self._behaviour(self.version, client)

def _fake_candidates(monkeypatch, behaviour, versions=("2026.09.01", "2026.08.19"),
                     refreshed=None):
    """Stub ytdlp.candidates; `refreshed` is what a refresh turns up."""
    seen = []
    def candidates(cfg, log, refresh=False):
        vs = refreshed if refresh and refreshed else versions
        return [FakeYtDlp(v, behaviour, seen) for v in vs]
    monkeypatch.setattr(acquire.ytdlp, "candidates", candidates)
    return seen

def test_download_falls_through_to_the_next_client(tmp_path, monkeypatch):
    from songtrace.ytdlp import YtDlpError
    def behaviour(version, client):
        if client is None:
            raise YtDlpError("ERROR: HTTP Error 403: Forbidden")
        (tmp_path / "source_video.webm").write_bytes(b"v")
    seen = _fake_candidates(monkeypatch, behaviour)
    cfg = Config(whisper_url="http://x/v1", ytdlp_clients=["default", "tv"])
    logged = []
    video, thumb = acquire.download_youtube("https://yt/x", tmp_path, cfg,
                                            logged.append)
    assert video.name == "source_video.webm"
    assert thumb is None
    assert seen == [("2026.09.01", None),
                    ("2026.09.01", "youtube:player_client=tv")]
    assert any("403" in m for m in logged)

def test_download_falls_back_to_the_bundled_copy(tmp_path, monkeypatch):
    from songtrace.ytdlp import YtDlpError
    def behaviour(version, client):
        if version == "2026.09.01":
            raise YtDlpError("rejected songtrace's options")
        (tmp_path / "source_video.webm").write_bytes(b"v")
    seen = _fake_candidates(monkeypatch, behaviour)
    cfg = Config(whisper_url="http://x/v1", ytdlp_clients=["default"])
    acquire.download_youtube("https://yt/x", tmp_path, cfg, lambda m: None)
    assert [v for v, _ in seen] == ["2026.09.01", "2026.08.19"]

def test_download_retries_with_a_release_found_after_failing(tmp_path, monkeypatch):
    from songtrace.ytdlp import YtDlpError
    def behaviour(version, client):
        if version != "2026.10.01":
            raise YtDlpError("ERROR: HTTP Error 403: Forbidden")
        (tmp_path / "source_video.webm").write_bytes(b"v")
    seen = _fake_candidates(monkeypatch, behaviour,
                            refreshed=("2026.10.01", "2026.08.19"))
    cfg = Config(whisper_url="http://x/v1", ytdlp_clients=["default"])
    acquire.download_youtube("https://yt/x", tmp_path, cfg, lambda m: None)
    assert [v for v, _ in seen] == ["2026.09.01", "2026.08.19", "2026.10.01"]

def test_download_reports_versions_and_clients_when_all_fail(tmp_path, monkeypatch):
    import pytest
    from songtrace.pipeline import PipelineError
    from songtrace.ytdlp import YtDlpError
    def behaviour(version, client):
        raise YtDlpError("ERROR: HTTP Error 403: Forbidden")
    seen = _fake_candidates(monkeypatch, behaviour)
    cfg = Config(whisper_url="http://x/v1", ytdlp_clients=["default", "tv"])
    with pytest.raises(PipelineError) as e:
        acquire.download_youtube("https://yt/x", tmp_path, cfg, lambda m: None)
    assert "2026.09.01, 2026.08.19" in str(e.value)
    assert "default, tv" in str(e.value)
    assert "403" in str(e.value)
    assert len(seen) == 4

def test_download_with_no_configured_clients_still_tries_default(tmp_path, monkeypatch):
    seen = _fake_candidates(
        monkeypatch,
        lambda v, c: (tmp_path / "source_video.mp4").write_bytes(b"v"))
    cfg = Config(whisper_url="http://x/v1", ytdlp_clients=[])
    acquire.download_youtube("https://yt/x", tmp_path, cfg, lambda m: None)
    assert seen == [("2026.09.01", None)]
