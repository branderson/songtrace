from pathlib import Path
from songtrace.ultrastar import (
    Note, LineBreak, parse_ultrastar, serialize_ultrastar,
    get_header, format_bpm, parse_bpm,
)

FIXTURES = Path(__file__).parent / "fixtures"

def load(name):
    return (FIXTURES / name).read_text(encoding="utf-8", errors="replace")

def test_parse_headers_and_notes():
    song = parse_ultrastar(load("sanctuary.txt"))
    assert get_header(song, "artist") == "Utada Hikaru"
    assert parse_bpm(get_header(song, "BPM")) == 217.98
    first = song.body[0]
    assert first == Note(kind=":", beat=0, length=3, pitch=16, text="ah")
    assert any(isinstance(x, LineBreak) and x.beat == 95 for x in song.body)
    assert any(n.kind == "F" for n in song.body if isinstance(n, Note))

def test_note_text_keeps_leading_space():
    song = parse_ultrastar(load("sanctuary.txt"))
    notes = [n for n in song.body if isinstance(n, Note)]
    assert notes[1].text == " ah"

def test_negative_pitch():
    song = parse_ultrastar(load("sweet_child.txt"))
    assert any(n.pitch == -9 for n in song.body if isinstance(n, Note))

def test_round_trip():
    for name in ("sweet_child.txt", "sanctuary.txt"):
        song = parse_ultrastar(load(name))
        again = parse_ultrastar(serialize_ultrastar(song))
        assert again == song

def test_format_bpm():
    assert format_bpm(256.0) == "256"
    assert format_bpm(217.98) == "217,98"
