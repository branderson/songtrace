import json
from pathlib import Path

from PIL import Image

from karaoke_maker.config import Config
from karaoke_maker.pipeline import STAGE_ORDER, Ctx, SongInput
from karaoke_maker.stages import all_stages, package
from karaoke_maker.ultrastar import get_header, parse_ultrastar

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


def test_package_copies_compatible_video(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "ffmpeg", lambda *a: Path(a[-1]).write_bytes(b"m"))
    monkeypatch.setattr(package, "ffprobe_codec", lambda p: ("webm", "vp9"))
    ctx = make_ctx(tmp_path, cover=make_cover(tmp_path))
    package.package_stage().run(ctx)
    out = tmp_path / "out" / "A - T"
    assert (out / "A - T.txt").exists()
    assert (out / "A - T.mp3").exists()
    assert (out / "A - T.webm").read_bytes() == b"v"
    img = Image.open(out / "A - T.jpg")
    assert img.width == img.height


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


def test_package_outputs_txt_and_mp3():
    ctx_outputs = package.package_stage().outputs
    cfg = Config(whisper_url="u", output_dir=Path("out"), work_dir=Path("work"))
    inp = SongInput(title="T", artist="A", lyrics_text="x")
    ctx = Ctx(inp=inp, cfg=cfg, work=Path("work"), log=lambda m: None)
    assert ctx_outputs(ctx) == [Path("out/A - T/A - T.txt"), Path("out/A - T/A - T.mp3")]


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
