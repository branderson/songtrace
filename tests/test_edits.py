import json

import pytest

from songtrace import edits

LYRICS = [["one", "two"], ["three", "four"], ["five", "six"]]


def gen(start, line, pitch=0, dur=0.4, low=False, text="la"):
    return {"start": start, "duration": dur, "pitch": pitch, "text": text,
            "low_confidence": low, "line_index": line}


def edited(*lines):
    data = edits._empty(LYRICS)
    for index, notes in lines:
        edits.set_line(data, index, notes)
    return data


def note(start, pitch=5, dur=0.4, text="la", kind=":"):
    return {"start": start, "duration": dur, "pitch": pitch, "text": text, "kind": kind}


GENERATED = [gen(1.0, 0), gen(1.5, 0), gen(4.0, 1), gen(4.5, 1), gen(8.0, 2), gen(8.5, 2)]
BREAKS = [3.0, 6.0]


def test_no_edits_leaves_the_generated_song_alone():
    assert edits.merge(GENERATED, BREAKS, edits._empty(LYRICS)) == (GENERATED, BREAKS)


def test_an_edited_line_replaces_its_generated_notes():
    notes, _ = edits.merge(GENERATED, BREAKS, edited(
        (1, [note(4.2, pitch=7, kind="*"), note(4.8, pitch=9)])))
    line1 = [n for n in notes if n["line_index"] == 1]
    assert [(n["start"], n["pitch"], n["kind"]) for n in line1] == [
        (4.2, 7, "*"), (4.8, 9, ":")]
    assert not any(n["low_confidence"] for n in line1)
    assert [n["line_index"] for n in notes] == [0, 0, 1, 1, 2, 2]


def test_neighbours_give_way_to_an_edited_line():
    # line 1 edited to run from 1.7s to 8.6s, over its neighbours' notes
    notes, _ = edits.merge(GENERATED, BREAKS, edited(
        (1, [note(1.7), note(8.2, dur=0.4)])))
    by_line = {}
    for n in notes:
        by_line.setdefault(n["line_index"], []).append(
            (round(n["start"], 2), round(n["start"] + n["duration"], 2)))
    assert by_line[0] == [(1.0, 1.4), (1.5, 1.7)]     # cut back to end at 1.7s
    assert by_line[2] == [(8.6, 8.9)]                 # 8.0 note gone, 8.5 starts at 8.6


def test_breaks_beside_an_edited_line_move_and_the_rest_stay():
    data = edited((2, [note(9.0), note(9.5)]))
    _, breaks = edits.merge(GENERATED, BREAKS, data)
    assert breaks[0] == 3.0                           # lines 0-1: generated
    assert breaks[1] == pytest.approx((4.9 + 9.0) / 2)  # lines 1-2: re-centred


def test_an_edit_follows_its_line_when_lines_are_added_above(tmp_path):
    data = edited((1, [note(4.0)]))
    edits.save(tmp_path, data)
    moved = edits.load(tmp_path, [["zero"]] + LYRICS)
    assert list(moved["lines"]) == ["2"] and moved["lines"]["2"]["line_index"] == 2
    assert moved["stale"] == []
    assert json.loads((tmp_path / edits.EDITS_FILE).read_text())["lyrics"][0] == ["zero"]


def test_an_edit_whose_words_changed_is_set_aside(tmp_path):
    edits.save(tmp_path, edited((1, [note(4.0)]), (2, [note(8.0)])))
    changed = [LYRICS[0], ["three", "more"], LYRICS[2]]
    data = edits.load(tmp_path, changed)
    assert list(data["lines"]) == ["2"]
    assert [s["words"] for s in data["stale"]] == [["three", "four"]]


def test_validate_notes_checks_and_orders():
    assert [n["start"] for n in edits.validate_notes([note(2.0), note(1.0)])] == [1.0, 2.0]
    for bad in (note(1.0, kind="X"), note(1.0, dur=0), note(-1.0), note(1.0, text=" "),
                {"start": 1.0}):
        with pytest.raises(ValueError):
            edits.validate_notes([bad])


def test_saving_no_edits_removes_the_file(tmp_path):
    edits.save(tmp_path, edited((0, [note(1.0)])))
    data = edits.load(tmp_path, LYRICS)
    edits.clear_line(data, 0)
    edits.save(tmp_path, data)
    assert not (tmp_path / edits.EDITS_FILE).exists()


def _write_generated(work, notes, warnings):
    (work / "notes_sec.json").write_text(json.dumps(
        {"notes": notes, "breaks": BREAKS, "warnings": warnings}))


def test_finalize_without_edits_passes_warnings_through(tmp_path):
    _write_generated(tmp_path, GENERATED, ["anything at all"])
    final = edits.finalize(tmp_path, LYRICS)
    assert final["warnings"] == ["anything at all"]
    assert final["edited_lines"] == [] and final["stale"] == []
    assert json.loads((tmp_path / edits.FINAL_FILE).read_text()) == final


def test_finalize_stops_reporting_a_line_the_user_fixed(tmp_path):
    notes = [gen(1.0, 0), gen(4.0, 1, low=True), gen(8.0, 2)]
    _write_generated(tmp_path, notes, ["1 low-confidence note near 0:04–0:04",
                                       "missing lyrics near 0:20–0:25"])
    edits.save(tmp_path, edited((1, [note(4.0)])))
    final = edits.finalize(tmp_path, LYRICS)
    assert final["warnings"] == ["missing lyrics near 0:20–0:25"]
    assert final["edited_lines"] == [1]


def test_finalize_reports_stale_edits(tmp_path):
    _write_generated(tmp_path, GENERATED, [])
    edits.save(tmp_path, edited((1, [note(4.0)])))
    final = edits.finalize(tmp_path, [LYRICS[0], ["other"], LYRICS[2]])
    assert final["warnings"] == ["1 edited line no longer matches the lyrics; "
                                 "set aside in the editor"]
