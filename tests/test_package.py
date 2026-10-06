import json
from pathlib import Path

import pytest
from PIL import Image

from songtrace.config import Config
from songtrace.pipeline import (
    STAGE_ORDER, Ctx, PipelineError, SongInput, run_pipeline, song_slug,
)
from songtrace.stages import all_stages, package
from songtrace.ultrastar import get_header, parse_ultrastar

SONG_TXT = ("#TITLE:T\n#ARTIST:A\n#MP3:A - T.mp3\n#COVER:A - T.jpg\n"
            "#VIDEO:A - T.webm\n#BPM:200\n#GAP:0\n: 0 1 0 la\nE\n")


def make_ctx(tmp_path, cover=None):
    work = tmp_path / "work"
    work.mkdir()
    (work / "song.txt").write_text(SONG_TXT)
    (work / "audio.wav").write_bytes(b"w")
    (work / "source_video.webm").write_bytes(b"v")
    (work / "meta.json").write_text(json.dumps(
        {"video_file": "source_video.webm", "thumbnail": None}))
    inp = SongInput(title="T", artist="A", lyrics_text="x", cover_path=cover)
    cfg = Config(whisper_url="u", output_dir=tmp_path / "out", work_dir=tmp_path)
    return Ctx(inp=inp, cfg=cfg, work=work, log=lambda m: None)


def make_cover(tmp_path):
    p = tmp_path / "c.png"
    Image.new("RGB", (640, 480), "red").save(p)
    return p


def make_pipeline_inputs(tmp_path, cover=None):
    """Set up inputs at the exact work-dir path run_pipeline() would use."""
    cfg = Config(whisper_url="u", output_dir=tmp_path / "out",
                 work_dir=tmp_path / "work_root")
    inp = SongInput(title="T", artist="A", lyrics_text="x", cover_path=cover)
    work = cfg.work_dir / song_slug(inp)
    work.mkdir(parents=True)
    (work / "song.txt").write_text(SONG_TXT)
    (work / "audio.wav").write_bytes(b"w")
    (work / "source_video.webm").write_bytes(b"v")
    (work / "meta.json").write_text(json.dumps(
        {"video_file": "source_video.webm", "thumbnail": None}))
    return inp, cfg


def test_package_remuxes_compatible_video_without_reencoding(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(package, "ffmpeg",
                        lambda *a: (calls.append(a), Path(a[-1]).write_bytes(b"m")))
    monkeypatch.setattr(package, "ffprobe_codec", lambda p: ("webm", "vp9"))
    ctx = make_ctx(tmp_path, cover=make_cover(tmp_path))
    package.package_stage().run(ctx)
    out = tmp_path / "out" / "A - T"
    assert (out / "A - T.txt").exists()
    assert (out / "A - T.mp3").exists()
    assert (out / "A - T.webm").exists()
    video_call = next(c for c in calls if c[-1].endswith("A - T.webm"))
    assert "-c" in video_call and "copy" in video_call, "the video was re-encoded"
    img = Image.open(out / "A - T.jpg")
    assert img.width == img.height


@pytest.mark.parametrize("probe,name", [(("webm", "vp9"), "A - T.webm"),
                                        (("avi", "mpeg4"), "A - T.mp4")])
def test_package_strips_the_videos_audio_track(tmp_path, monkeypatch, probe, name):
    """#MP3 is the song's audio; a video that keeps its own plays as an echo."""
    calls = []
    monkeypatch.setattr(package, "ffmpeg",
                        lambda *a: (calls.append(a), Path(a[-1]).write_bytes(b"m")))
    monkeypatch.setattr(package, "ffprobe_codec", lambda p: probe)
    ctx = make_ctx(tmp_path, cover=make_cover(tmp_path))
    package.package_stage().run(ctx)
    video_call = next(c for c in calls if c[-1].endswith(name))
    assert "-an" in video_call


def test_package_reencodes_and_rewrites_video_header(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "ffmpeg", lambda *a: Path(a[-1]).write_bytes(b"m"))
    monkeypatch.setattr(package, "ffprobe_codec", lambda p: ("avi", "mpeg4"))
    ctx = make_ctx(tmp_path, cover=make_cover(tmp_path))
    package.package_stage().run(ctx)
    out = tmp_path / "out" / "A - T"
    song = parse_ultrastar((out / "A - T.txt").read_text())
    assert get_header(song, "VIDEO") == "A - T.mp4"
    # VIDEO keeps its original slot, ahead of BPM/GAP
    keys = [k for k, _ in song.headers]
    assert keys.index("VIDEO") < keys.index("BPM") < keys.index("GAP")


def test_package_no_cover_drops_header(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "ffmpeg", lambda *a: Path(a[-1]).write_bytes(b"m"))
    monkeypatch.setattr(package, "ffprobe_codec", lambda p: ("webm", "vp9"))
    ctx = make_ctx(tmp_path, cover=None)
    package.package_stage().run(ctx)
    song = parse_ultrastar((tmp_path / "out" / "A - T" / "A - T.txt").read_text())
    assert get_header(song, "COVER") is None


def test_package_uses_work_thumbnail_when_no_cover_path(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "ffmpeg", lambda *a: Path(a[-1]).write_bytes(b"m"))
    monkeypatch.setattr(package, "ffprobe_codec", lambda p: ("webm", "vp9"))
    ctx = make_ctx(tmp_path, cover=None)
    make_cover(tmp_path).rename(ctx.work / "thumb.png")
    meta = json.loads((ctx.work / "meta.json").read_text())
    meta["thumbnail"] = "thumb.png"
    (ctx.work / "meta.json").write_text(json.dumps(meta))
    package.package_stage().run(ctx)
    out = tmp_path / "out" / "A - T"
    assert (out / "A - T.jpg").exists()
    song = parse_ultrastar((out / "A - T.txt").read_text())
    assert get_header(song, "COVER") == "A - T.jpg"


def test_package_outputs_done_marker():
    ctx_outputs = package.package_stage().outputs
    cfg = Config(whisper_url="u", output_dir=Path("out"), work_dir=Path("work"))
    inp = SongInput(title="T", artist="A", lyrics_text="x")
    ctx = Ctx(inp=inp, cfg=cfg, work=Path("work"), log=lambda m: None)
    assert ctx_outputs(ctx) == [Path("work/package.done")]


def test_package_corrupt_cover_degrades_to_no_cover(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "ffmpeg", lambda *a: Path(a[-1]).write_bytes(b"m"))
    monkeypatch.setattr(package, "ffprobe_codec", lambda p: ("webm", "vp9"))
    bad_cover = tmp_path / "bad.jpg"
    bad_cover.write_bytes(b"not an image")
    ctx = make_ctx(tmp_path, cover=bad_cover)
    package.package_stage().run(ctx)
    out = tmp_path / "out" / "A - T"
    assert not (out / "A - T.jpg").exists()
    song = parse_ultrastar((out / "A - T.txt").read_text())
    assert get_header(song, "COVER") is None


def test_package_partial_failure_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "ffmpeg", lambda *a: Path(a[-1]).write_bytes(b"m"))

    def boom(p):
        raise PipelineError("ffprobe failed")

    monkeypatch.setattr(package, "ffprobe_codec", boom)
    ctx = make_ctx(tmp_path, cover=make_cover(tmp_path))
    with pytest.raises(PipelineError):
        package.package_stage().run(ctx)
    out = tmp_path / "out" / "A - T"
    assert (out / "A - T.txt").exists()
    assert (out / "A - T.mp3").exists()
    assert not (out / "A - T.webm").exists()
    assert not package.package_stage().outputs(ctx)[0].exists()


def test_package_success_is_cached_by_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "ffmpeg", lambda *a: Path(a[-1]).write_bytes(b"m"))
    monkeypatch.setattr(package, "ffprobe_codec", lambda p: ("webm", "vp9"))
    inp, cfg = make_pipeline_inputs(tmp_path, cover=make_cover(tmp_path))
    stage = package.package_stage()
    run_pipeline(inp, cfg, log=lambda m: None, stages=[stage])
    logs = []
    run_pipeline(inp, cfg, log=logs.append, stages=[stage])
    assert any("[package] cached, skipping" in m for m in logs)


def test_package_retry_after_partial_failure_completes_folder(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky_ffprobe(p):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PipelineError("ffprobe failed")
        return ("webm", "vp9")

    monkeypatch.setattr(package, "ffmpeg", lambda *a: Path(a[-1]).write_bytes(b"m"))
    monkeypatch.setattr(package, "ffprobe_codec", flaky_ffprobe)
    inp, cfg = make_pipeline_inputs(tmp_path, cover=make_cover(tmp_path))
    stage = package.package_stage()
    with pytest.raises(PipelineError):
        run_pipeline(inp, cfg, log=lambda m: None, stages=[stage])
    out = cfg.output_dir / "A - T"
    assert (out / "A - T.txt").exists() and (out / "A - T.mp3").exists()
    assert not (out / "A - T.webm").exists()

    run_pipeline(inp, cfg, log=lambda m: None, stages=[stage])
    assert (out / "A - T.webm").exists()
    assert (out / "A - T.jpg").exists()


def test_slash_in_title_is_stripped_from_folder_and_filenames(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "ffmpeg", lambda *a: Path(a[-1]).write_bytes(b"m"))
    monkeypatch.setattr(package, "ffprobe_codec", lambda p: ("webm", "vp9"))
    ctx = make_ctx(tmp_path, cover=make_cover(tmp_path))
    ctx.inp.title = "T/2"
    package.package_stage().run(ctx)
    out = tmp_path / "out" / "A - T2"
    assert (out / "A - T2.txt").exists()
    assert (out / "A - T2.mp3").exists()


def test_all_stages_matches_stage_order():
    assert [s.name for s in all_stages()] == STAGE_ORDER
