import json

import numpy as np
import pytest
from songtrace.config import Config
from songtrace.notes import (
    MIN_NOTE_SEC, TimedNote, build_notes, hz_to_ultrastar, segment_span,
    syllabify,
)
from songtrace.pipeline import Ctx, SongInput
from songtrace.stages import all_stages, notes as notes_stage_mod


def flat_pitch(duration=10.0, hz=261.63):   # C4
    times = np.arange(0, duration, 0.01)
    return times, np.full_like(times, hz), np.full_like(times, 0.9)


def assert_no_overlap(notes):
    for a, b in zip(notes, notes[1:]):
        assert a.start < b.start
        assert a.start + a.duration <= b.start
    assert all(n.duration > 0 for n in notes)


def test_hz_to_ultrastar():
    assert hz_to_ultrastar(261.63) == 0        # C4
    assert hz_to_ultrastar(440.0) == 9         # A4
    assert hz_to_ultrastar(220.0) == -3        # A3


def test_hz_to_ultrastar_rejects_non_positive():
    with pytest.raises(ValueError):
        hz_to_ultrastar(0.0)
    with pytest.raises(ValueError):
        hz_to_ultrastar(-100.0)


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


def test_syllabify_language_without_pyphen_dict_is_not_mishandled_as_english():
    # Japanese is a known language (e.g. to whisper_client) but has no pyphen
    # dictionary; it must not silently fall back to English hyphenation rules.
    assert syllabify("beautiful", "Japanese") == ["beautiful"]
    assert syllabify("beautiful", "English") == ["beau", "ti", "ful"]


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
    assert warnings[0] == "1 low-confidence note near 0:01–0:01"


def test_build_notes_low_score_word_flagged():
    times, f0, conf = flat_pitch()
    rows = [{"word": "hey", "start": 1.0, "end": 1.4, "score": 0.1, "line_index": 0}]
    notes, _, warnings = build_notes(rows, times, f0, conf, "English")
    assert all(n.low_confidence for n in notes) and len(warnings) == 1


def test_build_notes_unvoiced_word_inherits_previous_pitch():
    times, f0, conf = flat_pitch(hz=440.0)      # A4 -> 9, not the fallback 0
    conf[300:] = 0.0
    rows = [
        {"word": "one", "start": 1.0, "end": 2.0, "score": 0.9, "line_index": 0},
        {"word": "two", "start": 4.0, "end": 4.5, "score": 0.9, "line_index": 0},
    ]
    notes, _, _ = build_notes(rows, times, f0, conf, "English")
    assert notes[1].pitch == notes[0].pitch == 9
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


def test_build_notes_subdivides_the_widest_segment_first():
    times = np.arange(0, 2.0, 0.01)
    f0 = np.where(times < 1.1, 329.63, 261.63)   # 0.1s E4 then 0.9s C4 in-word
    conf = np.full_like(times, 0.9)
    rows = [{"word": "beautiful", "start": 1.0, "end": 2.0, "score": 0.9,
             "line_index": 0}]
    notes, _, _ = build_notes(rows, times, f0, conf, "English")
    assert [n.text for n in notes] == ["beau", "ti", "ful"]
    assert [n.pitch for n in notes] == [4, 0, 0]      # the 0.9s segment split
    assert abs(notes[0].duration - 0.1) < 1e-6
    assert abs(notes[1].duration - notes[2].duration) < 1e-6


def test_build_notes_zero_duration_row_gets_minimum_length():
    times, f0, conf = flat_pitch()
    rows = [
        {"word": "...", "start": 1.0, "end": 1.0, "score": 0.0, "line_index": 0},
        {"word": "go", "start": 1.0, "end": 1.5, "score": 0.9, "line_index": 0},
    ]
    notes, _, _ = build_notes(rows, times, f0, conf, "English")
    assert notes[0].duration >= MIN_NOTE_SEC
    assert_no_overlap(notes)


def test_build_notes_repeated_zero_duration_rows_stay_separate():
    times, f0, conf = flat_pitch()
    rows = [{"word": w, "start": 1.0, "end": 1.0, "score": 0.0, "line_index": 0}
            for w in ("...", "...", "...", "...")]
    rows.append({"word": "go", "start": 1.0, "end": 1.5, "score": 0.9,
                 "line_index": 0})
    notes, _, _ = build_notes(rows, times, f0, conf, "English")
    assert len(notes) == 5
    assert_no_overlap(notes)


def test_build_notes_short_multisyllable_word_does_not_overlap():
    times, f0, conf = flat_pitch()
    rows = [
        {"word": "reminds", "start": 8.92, "end": 8.99, "score": 0.9,
         "line_index": 0},
        {"word": "me", "start": 8.99, "end": 9.20, "score": 0.9, "line_index": 0},
    ]
    notes, _, _ = build_notes(rows, times, f0, conf, "English")
    assert [n.text for n in notes] == ["re", "minds", " me"]
    assert_no_overlap(notes)


def test_build_notes_never_overlaps_on_dense_input():
    rng = np.random.default_rng(11)
    times = np.arange(0, 40.0, 0.01)
    words = ["We", "count", "a", "beautiful", "number", "reminds", "us",
             "everything", "we", "listed", "yesterday", "oh"]
    for _ in range(8):
        f0 = 220.0 * 2 ** (rng.integers(0, 12, len(times)) // 3 / 12.0)
        conf = np.where(rng.random(len(times)) < 0.9, 0.9, 0.1)
        rows, t = [], 0.5
        for i in range(80):
            dur = float(rng.uniform(0.06, 0.55))
            rows.append({"word": words[i % len(words)], "start": t,
                         "end": t + dur, "score": 0.9, "line_index": i // 6})
            t += dur + float(rng.uniform(0.0, 0.15))
        notes, _, _ = build_notes(rows, times, f0, conf, "English")
        assert_no_overlap(notes)


def test_build_notes_starts_are_non_decreasing():
    times = np.arange(0, 12.0, 0.01)
    f0 = 220.0 * 2 ** (np.floor(times * 2) % 5 / 12.0)   # stepped melody
    conf = np.where((times % 3.0) < 2.6, 0.9, 0.1)       # unvoiced gaps
    words = ["We", "count", "a", "beautiful", "number", "that",
             "reminds", "us", "of", "everything", "we", "list"]
    rows = []
    for i, w in enumerate(words):
        dur = 0.8 if i % 3 else 0.07       # every third word sung very fast
        rows.append({"word": w, "start": 0.5 + i * 0.9, "end": 0.5 + i * 0.9 + dur,
                     "score": 0.9 if i % 5 else 0.2, "line_index": i // 4})
    notes, breaks, warnings = build_notes(rows, times, f0, conf, "English")
    assert len(notes) >= len(words)
    starts = [n.start for n in notes]
    assert starts == sorted(starts)
    assert breaks == sorted(breaks) and len(breaks) == 2
    assert_no_overlap(notes)
    assert warnings and all("low-confidence note" in w for w in warnings)


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
    assert "notes" in [s.name for s in all_stages()]


def test_build_notes_skips_notes_within_missing_span():
    times, f0, conf = flat_pitch()
    rows = [
        {"word": "keep", "start": 1.0, "end": 1.4, "score": 0.9, "line_index": 0},
        {"word": "skip", "start": 5.0, "end": 5.4, "score": 0.9, "line_index": 1},
    ]
    notes, _, _ = build_notes(rows, times, f0, conf, "English",
                              missing_spans=[(4.5, 6.0)])
    assert [n.text for n in notes] == ["keep"]


def test_warnings_report_dropped_lines():
    times, f0, conf = flat_pitch()
    rows = [{"word": "keep", "start": 1.0, "end": 1.4, "score": 0.9, "line_index": 0}]
    ref_lines = [["keep"], ["the", "outro"]]
    _, _, warnings = build_notes(rows, times, f0, conf, "English",
                                 dropped_lines=[1], ref_lines=ref_lines)
    assert warnings == ['1 line dropped (not sung) near "the outro"']


def test_warnings_report_missing_spans():
    times, f0, conf = flat_pitch()
    rows = [{"word": "keep", "start": 1.0, "end": 1.4, "score": 0.9, "line_index": 0}]
    _, _, warnings = build_notes(rows, times, f0, conf, "English",
                                 missing_spans=[(81.0, 89.5)])
    assert warnings == ["missing lyrics near 1:21–1:29"]


def test_build_notes_skipped_row_mid_line_does_not_corrupt_spacing_or_breaks():
    # "gone" falls inside a missing span and is skipped; "away" is the same
    # line's next word and must still get its leading space, and the break
    # before the next line must use the last *emitted* word's end (3.4, from
    # "away"), not the skipped row's end (2.4).
    times, f0, conf = flat_pitch()
    rows = [
        {"word": "keep", "start": 1.0, "end": 1.4, "score": 0.9, "line_index": 0},
        {"word": "gone", "start": 2.0, "end": 2.4, "score": 0.9, "line_index": 0},
        {"word": "away", "start": 3.0, "end": 3.4, "score": 0.9, "line_index": 0},
        {"word": "next", "start": 5.0, "end": 5.4, "score": 0.9, "line_index": 1},
    ]
    notes, breaks, _ = build_notes(rows, times, f0, conf, "English",
                                   missing_spans=[(1.9, 2.5)])
    assert [n.text for n in notes] == ["keep", " away", "next"]
    assert breaks == [(3.4 + 5.0) / 2]

def test_notes_carry_line_index():
    times, f0, conf = flat_pitch()
    rows = [{"word": "hey", "start": 1.0, "end": 1.4, "score": 0.9, "line_index": 3}]
    notes, _, _ = build_notes(rows, times, f0, conf, "English")
    assert notes[0].line_index == 3
