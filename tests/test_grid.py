import json

import numpy as np
import pytest
from karaoke_maker.config import Config
from karaoke_maker.grid import scale_bpm, to_ultrastar
from karaoke_maker.pipeline import Ctx, SongInput
from karaoke_maker.stages import all_stages, grid as grid_stage_mod
from karaoke_maker.ultrastar import (
    LineBreak, Note, get_header, parse_ultrastar,
)


def note(start, duration, pitch=0, text="la"):
    return {"start": start, "duration": duration, "pitch": pitch, "text": text,
            "low_confidence": False}


def assert_ordered(body):
    """Beats never go backwards and no two notes share a beat."""
    cursor = None
    for item in body:
        if cursor is not None:
            assert item.beat >= cursor
        cursor = item.beat + (0 if isinstance(item, LineBreak) else item.length)
    notes = [i for i in body if isinstance(i, Note)]
    for a, b in zip(notes, notes[1:]):
        assert a.beat + a.length <= b.beat
    assert all(n.length >= 1 for n in notes)


def test_scale_bpm():
    assert scale_bpm(128.0) == 256.0
    assert scale_bpm(64.0) == 256.0
    assert scale_bpm(250.0) == 250.0
    assert scale_bpm(450.0) == 225.0


@pytest.mark.parametrize("bpm", [0.0, -120.0, float("inf"), float("nan")])
def test_scale_bpm_rejects_unusable_tempo(bpm):
    with pytest.raises(ValueError):
        scale_bpm(bpm)


def test_to_ultrastar_rejects_empty_notes():
    with pytest.raises(ValueError):
        to_ultrastar([], [], 240.0, [])


def test_to_ultrastar_conversion():
    notes = [note(2.0, 0.5, 3, "la"), note(2.6, 0.25, 5, " la")]
    song = to_ultrastar(notes, breaks=[2.55], file_bpm=240.0,
                        headers=[("TITLE", "T")])
    # spb = 60/(4*240) = 0.0625s; gap = 2000ms
    assert get_header(song, "GAP") == "2000"
    assert get_header(song, "BPM") == "240"
    assert song.headers[0] == ("TITLE", "T")
    n0, br, n1 = song.body
    assert n0 == Note(kind=":", beat=0, length=8, pitch=3, text="la")
    assert br == LineBreak(beat=9)          # (2.55-2.0)/0.0625 = 8.8 -> 9
    assert n1.beat == 10 and n1.length == 4


def test_min_length_one():
    song = to_ultrastar([note(1.0, 0.001, 0, "x")], [], 240.0, [])
    assert song.body[0].length == 1


def test_fractional_bpm_header():
    song = to_ultrastar([note(0.0, 1.0)], [], 232.5, [])
    assert get_header(song, "BPM") == "232,50"


def test_beat_cursor_separates_notes_quantised_onto_one_beat():
    notes = [note(1.0, 0.01, 0, "a"), note(1.01, 0.01, 1, "b"),
             note(1.02, 0.01, 2, "c")]
    song = to_ultrastar(notes, [], 240.0, [])   # all three round to beat 0
    assert [n.beat for n in song.body] == [0, 1, 2]
    assert [n.length for n in song.body] == [1, 1, 1]
    assert_ordered(song.body)


def test_beat_cursor_preserves_onsets_that_do_not_collide():
    notes = [note(0.0, 0.5), note(1.0, 0.5), note(2.0, 0.5)]
    song = to_ultrastar(notes, [], 240.0, [])
    assert [n.beat for n in song.body] == [0, 16, 32]


def test_line_break_pushed_past_the_preceding_note():
    notes = [note(0.0, 0.56), note(0.57, 0.1)]
    song = to_ultrastar(notes, [0.51], 240.0, [])   # break rounds to beat 8
    n0, br, n1 = song.body
    assert n0.beat + n0.length == 9
    assert br == LineBreak(beat=9)
    assert n1.beat >= br.beat
    assert_ordered(song.body)


def test_line_break_on_a_note_start_precedes_that_note():
    """No gap across a line boundary puts the break exactly on the next onset."""
    notes = [note(0.0, 0.5, 0, "end"), note(0.5, 0.5, 2, "New")]
    song = to_ultrastar(notes, [0.5], 240.0, [])
    assert song.body[1] == LineBreak(beat=8)
    assert [n.text for n in song.body if isinstance(n, Note)] == ["end", "New"]
    assert_ordered(song.body)


def test_line_break_not_after_the_following_note():
    notes = [note(0.0, 0.02, 0, "a"), note(0.03, 0.02, 1, "b")]
    song = to_ultrastar(notes, [0.025], 240.0, [])
    br = song.body[1]
    assert isinstance(br, LineBreak)
    assert br.beat <= song.body[2].beat
    assert_ordered(song.body)


def sung_words(rng):
    """Words cut into equal syllable slices, the way build_notes emits them."""
    notes, breaks, t = [], [], 1.7
    for i in range(60):
        word = float(rng.uniform(0.05, 0.6))
        sylls = int(rng.integers(1, 4))
        for k in range(sylls):
            notes.append(note(t + k * word / sylls, word / sylls,
                              int(rng.integers(-12, 12))))
        t += word + float(rng.uniform(0.0, 0.4))
        if i % 5 == 4:
            breaks.append(t - 0.05)
    return notes, breaks


@pytest.mark.parametrize("bpm", [200.0, 300.0, 399.0])
def test_no_overlap_on_realistic_input(bpm):
    rng = np.random.default_rng(7)
    moved = 0
    for _ in range(20):
        notes, breaks = sung_words(rng)
        song = to_ultrastar(notes, breaks, bpm, [])
        assert len(song.body) == len(notes) + len(breaks)
        assert song.body[0].beat == 0
        assert_ordered(song.body)
        spb = 60 / (4 * bpm)
        emitted = [n for n in song.body if isinstance(n, Note)]
        moved += sum(1 for n, src in zip(emitted, notes)
                     if n.beat != round((src["start"] - notes[0]["start"]) / spb))
    assert moved > 0        # otherwise the fuzz never exercises the beat cursor


def write_work(work, video="source_video.webm"):
    work.mkdir()
    (work / "notes_sec.json").write_text(json.dumps({
        "notes": [note(1.0, 0.5, 0, "la"), note(2.0, 0.5, 2, "la")],
        "breaks": [1.75],
        "warnings": [],
    }))
    (work / "meta.json").write_text(json.dumps(
        {"video_file": video, "thumbnail": "source_video.jpg"}))


def make_ctx(work, **kwargs):
    inp = SongInput(title="Song", artist="Band", lyrics_text="x", **kwargs)
    return Ctx(inp=inp, cfg=Config(whisper_url="u", creator="me"), work=work,
               log=lambda m: None)


def test_grid_stage_writes_song_txt(tmp_path, monkeypatch):
    work = tmp_path / "w"
    write_work(work)
    monkeypatch.setattr(grid_stage_mod.grid, "detect_bpm", lambda p: 120.0)
    ctx = make_ctx(work, language="English", genre="Pop", year=1999)
    stage = grid_stage_mod.grid_stage()
    assert stage.outputs(ctx) == [work / "song.txt"]
    stage.run(ctx)
    song = parse_ultrastar((work / "song.txt").read_text(encoding="utf-8"))
    assert [k for k, _ in song.headers] == [
        "TITLE", "ARTIST", "LANGUAGE", "GENRE", "YEAR", "CREATOR",
        "MP3", "COVER", "VIDEO", "BPM", "GAP"]
    assert get_header(song, "BPM") == "240"        # 120 doubled into range
    assert get_header(song, "GAP") == "1000"
    assert get_header(song, "MP3") == "Band - Song.mp3"
    assert get_header(song, "COVER") == "Band - Song.jpg"
    assert get_header(song, "VIDEO") == "Band - Song.webm"
    assert [n.text for n in song.body if isinstance(n, Note)] == ["la", "la"]
    assert_ordered(song.body)


def test_grid_stage_omits_optional_headers(tmp_path, monkeypatch):
    work = tmp_path / "w"
    write_work(work, video="source_video.mp4")
    monkeypatch.setattr(grid_stage_mod.grid, "detect_bpm", lambda p: 210.0)
    ctx = make_ctx(work)
    grid_stage_mod.grid_stage().run(ctx)
    song = parse_ultrastar((work / "song.txt").read_text(encoding="utf-8"))
    assert [k for k, _ in song.headers] == [
        "TITLE", "ARTIST", "CREATOR", "MP3", "COVER", "VIDEO", "BPM", "GAP"]
    assert get_header(song, "VIDEO") == "Band - Song.mp4"
    assert get_header(song, "CREATOR") == "me"


def test_grid_stage_uses_bpm_override(tmp_path, monkeypatch):
    work = tmp_path / "w"
    write_work(work)

    def boom(path):
        raise AssertionError("detect_bpm must not run with an override")

    monkeypatch.setattr(grid_stage_mod.grid, "detect_bpm", boom)
    ctx = make_ctx(work, bpm_override=333.0)
    grid_stage_mod.grid_stage().run(ctx)
    song = parse_ultrastar((work / "song.txt").read_text(encoding="utf-8"))
    assert get_header(song, "BPM") == "333"


def test_grid_stage_registered():
    names = [s.name for s in all_stages()]
    assert names[-1] == "grid" and names[-2] == "notes"
