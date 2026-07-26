import json

import numpy as np
from karaoke_maker.config import Config
from karaoke_maker.notes import (
    MIN_NOTE_SEC, TimedNote, build_notes, hz_to_ultrastar, segment_span,
    syllabify,
)
from karaoke_maker.pipeline import Ctx, SongInput
from karaoke_maker.stages import all_stages, notes as notes_stage_mod


def flat_pitch(duration=10.0, hz=261.63):   # C4
    times = np.arange(0, duration, 0.01)
    return times, np.full_like(times, hz), np.full_like(times, 0.9)


def test_hz_to_ultrastar():
    assert hz_to_ultrastar(261.63) == 0        # C4
    assert hz_to_ultrastar(440.0) == 9         # A4
    assert hz_to_ultrastar(220.0) == -3        # A3


def test_segment_span_single_flat():
    times, f0, conf = flat_pitch()
    segs = segment_span(times, f0, conf, 1.0, 2.0)
    assert len(segs) == 1
    s, e, midi = segs[0]
    assert abs(midi - 60.0) < 0.1
    assert abs(s - 1.0) < 0.02 and abs(e - 2.0) < 0.02


def test_segment_span_splits_on_jump():
    times = np.arange(0, 2.0, 0.01)
    f0 = np.where(times < 1.0, 261.63, 329.63)   # C4 -> E4
    conf = np.full_like(times, 0.9)
    segs = segment_span(times, f0, conf, 0.0, 2.0)
    assert len(segs) == 2
    assert abs(segs[0][2] - 60.0) < 0.2 and abs(segs[1][2] - 64.0) < 0.2
    assert abs(segs[0][1] - segs[1][0]) < 1e-6      # contiguous


def test_segment_span_masks_low_conf():
    times, f0, conf = flat_pitch()
    conf[:] = 0.1
    assert segment_span(times, f0, conf, 0.0, 1.0) == []


def test_segment_span_ignores_brief_pitch_glitch():
    times, f0, conf = flat_pitch(duration=1.0)
    f0[50:52] = 523.25                          # 2-frame octave blip
    conf = np.full_like(times, 0.9)
    segs = segment_span(times, f0, conf, 0.0, 1.0)
    assert len(segs) == 1


def test_segment_span_drops_sub_minimum_segments():
    times, f0, conf = flat_pitch(duration=1.0)
    assert segment_span(times, f0, conf, 0.5, 0.52) == []


def test_syllabify():
    assert syllabify("reminds", "English") == ["re", "minds"]
    assert syllabify("a", "English") == ["a"]
    assert syllabify("xyzzy", None) == ["xyzzy"]
    assert syllabify("She's", "English") == ["She's"]


def test_syllabify_preserves_word_characters():
    assert "".join(syllabify("well-known", "English")) == "well-known"
    assert "".join(syllabify("beautiful", "English")) == "beautiful"


def test_build_notes_basic_line():
    times, f0, conf = flat_pitch()
    rows = [
        {"word": "She's", "start": 1.0, "end": 1.5, "score": 0.9, "line_index": 0},
        {"word": "got", "start": 1.6, "end": 2.0, "score": 0.9, "line_index": 0},
        {"word": "Now", "start": 5.0, "end": 5.5, "score": 0.9, "line_index": 1},
    ]
    notes, breaks, warnings = build_notes(rows, times, f0, conf, "English")
    assert notes[0].text == "She's"          # line-initial: no leading space
    assert notes[1].text == " got"           # mid-line: leading space
    assert notes[2].text == "Now"            # new line: no leading space
    assert len(breaks) == 1 and 2.0 <= breaks[0] <= 5.0
    assert warnings == []


def test_build_notes_melisma_continuation():
    times = np.arange(0, 3.0, 0.01)
    f0 = np.where(times < 1.0, 261.63, 329.63)
    conf = np.full_like(times, 0.9)
    rows = [{"word": "mine", "start": 0.2, "end": 2.8, "score": 0.9, "line_index": 0}]
    notes, _, _ = build_notes(rows, times, f0, conf, "English")
    assert [n.text for n in notes] == ["mine", "~"]
    assert notes[0].pitch != notes[1].pitch


def test_build_notes_unvoiced_word_flagged():
    times, f0, conf = flat_pitch()
    conf[:] = 0.0
    rows = [{"word": "hey", "start": 1.0, "end": 1.4, "score": 0.9, "line_index": 0}]
    notes, _, warnings = build_notes(rows, times, f0, conf, "English")
    assert notes[0].low_confidence and len(warnings) == 1
    assert warnings[0] == "1 low-confidence notes near 0:01–0:01"


def test_build_notes_low_score_word_flagged():
    times, f0, conf = flat_pitch()
    rows = [{"word": "hey", "start": 1.0, "end": 1.4, "score": 0.1, "line_index": 0}]
    notes, _, warnings = build_notes(rows, times, f0, conf, "English")
    assert all(n.low_confidence for n in notes) and len(warnings) == 1


def test_build_notes_unvoiced_word_inherits_previous_pitch():
    times, f0, conf = flat_pitch()
    conf[300:] = 0.0
    rows = [
        {"word": "one", "start": 1.0, "end": 2.0, "score": 0.9, "line_index": 0},
        {"word": "two", "start": 4.0, "end": 4.5, "score": 0.9, "line_index": 0},
    ]
    notes, _, _ = build_notes(rows, times, f0, conf, "English")
    assert notes[1].pitch == notes[0].pitch == 0
    assert notes[1].low_confidence and not notes[0].low_confidence


def test_build_notes_unvoiced_first_word_uses_zero_pitch():
    times, f0, conf = flat_pitch()
    conf[:] = 0.0
    rows = [{"word": "hey", "start": 1.0, "end": 1.4, "score": 0.9, "line_index": 0}]
    notes, _, _ = build_notes(rows, times, f0, conf, "English")
    assert notes[0].pitch == 0


def test_build_notes_multisyllable_word_emits_syllables_only():
    times, f0, conf = flat_pitch()
    rows = [
        {"word": "reminds", "start": 1.0, "end": 1.8, "score": 0.9, "line_index": 0},
        {"word": "reminds", "start": 2.0, "end": 2.8, "score": 0.9, "line_index": 0},
    ]
    notes, _, _ = build_notes(rows, times, f0, conf, "English")
    assert [n.text for n in notes] == ["re", "minds", " re", "minds"]


def test_build_notes_more_syllables_than_segments():
    times, f0, conf = flat_pitch()
    rows = [{"word": "beautiful", "start": 1.0, "end": 1.9, "score": 0.9,
             "line_index": 0}]
    notes, _, _ = build_notes(rows, times, f0, conf, "English")
    assert [n.text for n in notes] == ["beau", "ti", "ful"]
    assert all(abs(n.duration - notes[0].duration) < 1e-6 for n in notes)
    for a, b in zip(notes, notes[1:]):
        assert abs(a.start + a.duration - b.start) < 1e-6


def test_build_notes_zero_duration_row_gets_minimum_length():
    times, f0, conf = flat_pitch()
    rows = [
        {"word": "...", "start": 1.0, "end": 1.0, "score": 0.0, "line_index": 0},
        {"word": "go", "start": 1.0, "end": 1.5, "score": 0.9, "line_index": 0},
    ]
    notes, _, _ = build_notes(rows, times, f0, conf, "English")
    assert notes[0].duration >= MIN_NOTE_SEC
    assert all(n.duration > 0 for n in notes)


def test_build_notes_starts_are_non_decreasing():
    times = np.arange(0, 12.0, 0.01)
    f0 = 220.0 * 2 ** (np.floor(times * 2) % 5 / 12.0)   # stepped melody
    conf = np.where((times % 3.0) < 2.6, 0.9, 0.1)       # unvoiced gaps
    words = ["We", "count", "a", "beautiful", "number", "that",
             "reminds", "us", "of", "everything", "we", "list"]
    rows = []
    for i, w in enumerate(words):
        rows.append({"word": w, "start": 0.5 + i * 0.9, "end": 0.5 + i * 0.9 + 0.8,
                     "score": 0.9 if i % 5 else 0.2, "line_index": i // 4})
    notes, breaks, warnings = build_notes(rows, times, f0, conf, "English")
    assert len(notes) >= len(words)
    starts = [n.start for n in notes]
    assert starts == sorted(starts)
    assert breaks == sorted(breaks) and len(breaks) == 2
    assert all(n.duration > 0 for n in notes)
    assert warnings and all("low-confidence notes near" in w for w in warnings)


def test_build_notes_line_initial_after_break_has_no_space():
    times, f0, conf = flat_pitch()
    rows = [
        {"word": "one", "start": 1.0, "end": 1.4, "score": 0.9, "line_index": 0},
        {"word": "two", "start": 2.0, "end": 2.4, "score": 0.9, "line_index": 1},
        {"word": "three", "start": 3.0, "end": 3.4, "score": 0.9, "line_index": 1},
    ]
    notes, breaks, _ = build_notes(rows, times, f0, conf, "English")
    assert [n.text for n in notes] == ["one", "two", " three"]
    assert breaks == [1.7]


def test_notes_stage_writes_json(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    times, f0, conf = flat_pitch()
    np.savez(work / "pitch.npz", times=times, f0=f0, conf=conf)
    (work / "alignment.json").write_text(json.dumps([
        {"word": "She's", "start": 1.0, "end": 1.5, "score": 0.9, "line_index": 0},
        {"word": "gone", "start": 3.0, "end": 3.5, "score": 0.9, "line_index": 1},
    ]))
    ctx = Ctx(inp=SongInput(title="T", artist="A", lyrics_text="x", language="English"),
              cfg=Config(whisper_url="u"), work=work, log=lambda m: None)
    stage = notes_stage_mod.notes_stage()
    assert stage.outputs(ctx) == [work / "notes_sec.json"]
    stage.run(ctx)
    data = json.loads((work / "notes_sec.json").read_text())
    assert [n["text"] for n in data["notes"]] == ["She's", "gone"]
    assert data["breaks"] == [2.25]
    assert data["warnings"] == []
    assert TimedNote(**data["notes"][0]).pitch == 0


def test_notes_stage_registered():
    assert [s.name for s in all_stages()][-1] == "notes"
