import json

import pytest

from songtrace.config import Config
from songtrace.lyrics import normalize_lyrics
from songtrace.pipeline import Ctx, PipelineError, SongInput
from songtrace.stages import align

LYRICS = "it's one two three\nfour five six seven eight"


def make_ctx(tmp_path, lyrics=LYRICS, transcript=None):
    work = tmp_path / "w"
    work.mkdir(exist_ok=True)
    if transcript is None:
        transcript = [{"start": 1.0, "end": 4.0,
                       "text": "it's one two three four five six seven eight"}]
    (work / "transcript.json").write_text(json.dumps(transcript))
    inp = SongInput(title="T", artist="A", lyrics_text=lyrics)
    return Ctx(inp=inp, cfg=Config(whisper_url="u", device="cpu"),
               work=work, log=lambda m: None)


def fake_align(vocals_path, start, end, tokens, device):
    step = (end - start) / len(tokens)
    return [(start + i * step, start + (i + 1) * step, 0.9)
            for i in range(len(tokens))]


def run_align(tmp_path, monkeypatch, lyrics=LYRICS, transcript=None):
    monkeypatch.setattr(align, "force_align_section", fake_align)
    ctx = make_ctx(tmp_path, lyrics, transcript)
    align.align_stage().run(ctx)
    return json.loads((ctx.work / "alignment.json").read_text())


def check_invariants(rows, lyrics):
    ref = normalize_lyrics(lyrics)
    assert [r["word"] for r in rows] == ref.words
    assert [r["line_index"] for r in rows] == ref.word_line_index
    prev = 0.0
    for r in rows:
        assert r["start"] is not None and r["end"] is not None
        assert prev <= r["start"] <= r["end"]
        prev = r["end"]


def test_alignment_covers_all_words(tmp_path, monkeypatch):
    rows = run_align(tmp_path, monkeypatch)
    assert [r["word"] for r in rows] == \
        ["it's", "one", "two", "three", "four", "five", "six", "seven", "eight"]
    assert rows[0]["line_index"] == 0 and rows[-1]["line_index"] == 1
    starts = [r["start"] for r in rows]
    assert starts == sorted(starts)


def test_mismatched_lyrics_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(align, "force_align_section", fake_align)
    ctx = make_ctx(tmp_path, lyrics="totally unrelated nonsense words banana")
    with pytest.raises(PipelineError, match="similarity"):
        align.align_stage().run(ctx)


def test_unalignable_word_interpolated(tmp_path, monkeypatch):
    lyrics = "it's one ... three\nfour five six seven eight"
    rows = run_align(tmp_path, monkeypatch, lyrics)
    dots = next(r for r in rows if r["word"] == "...")
    assert dots["score"] == 0.0
    prev = next(r for r in rows if r["word"] == "one")
    nxt = next(r for r in rows if r["word"] == "three")
    assert prev["end"] <= dots["start"] <= dots["end"] <= nxt["start"] + 1e-6
    check_invariants(rows, lyrics)


def test_leading_unalignable_word(tmp_path, monkeypatch):
    lyrics = "... one two three\nfour five six seven eight"
    rows = run_align(tmp_path, monkeypatch, lyrics)
    assert rows[0]["word"] == "..." and rows[0]["score"] == 0.0
    assert rows[0]["start"] >= 0.0
    assert rows[0]["end"] <= rows[1]["start"] + 1e-6
    check_invariants(rows, lyrics)


def test_trailing_unalignable_word(tmp_path, monkeypatch):
    lyrics = "it's one two three\nfour five six seven ..."
    rows = run_align(tmp_path, monkeypatch, lyrics)
    assert rows[-1]["word"] == "..." and rows[-1]["score"] == 0.0
    assert rows[-2]["end"] <= rows[-1]["start"]
    check_invariants(rows, lyrics)


def test_consecutive_unalignable_words(tmp_path, monkeypatch):
    lyrics = "it's one ... ... three\nfour five six seven eight"
    rows = run_align(tmp_path, monkeypatch, lyrics)
    dots = [r for r in rows if r["word"] == "..."]
    assert len(dots) == 2
    assert all(r["score"] == 0.0 for r in dots)
    assert dots[0]["end"] <= dots[1]["start"]
    check_invariants(rows, lyrics)


def test_all_words_unalignable_in_section(tmp_path, monkeypatch):
    lyrics = "... ...\n... ..."
    transcript = [{"start": 1.0, "end": 4.0, "text": "..."}]
    monkeypatch.setattr(align, "force_align_section", fake_align)
    ctx = make_ctx(tmp_path, lyrics, transcript)
    align.align_stage().run(ctx)
    rows = json.loads((ctx.work / "alignment.json").read_text())
    check_invariants(rows, lyrics)
    assert all(r["score"] == 0.0 for r in rows)
    assert rows[-1]["end"] <= 5.0


def test_non_chronological_transcript(tmp_path, monkeypatch):
    transcript = [{"start": 30.0, "end": 33.0, "text": "four five six seven eight"},
                  {"start": 1.0, "end": 4.0, "text": "it's one two three"}]
    rows = run_align(tmp_path, monkeypatch, transcript=transcript)
    check_invariants(rows, LYRICS)
    assert len({r["start"] for r in rows}) == len(rows)
    assert rows[3]["end"] <= 5.0
    assert rows[4]["start"] >= 29.0


def test_inverted_section_does_not_collapse(tmp_path, monkeypatch):
    monkeypatch.setattr(align, "vocal_sections", lambda segments: [(29.0, 5.0)])
    rows = run_align(tmp_path, monkeypatch)
    check_invariants(rows, LYRICS)
    assert len({r["start"] for r in rows}) == len(rows)
    assert rows[0]["start"] >= 5.0 and rows[-1]["end"] <= 29.0


def test_words_split_across_sections(tmp_path, monkeypatch):
    transcript = [{"start": 1.0, "end": 4.0, "text": "it's one two three"},
                  {"start": 30.0, "end": 33.0, "text": "four five six seven eight"}]
    rows = run_align(tmp_path, monkeypatch, transcript=transcript)
    check_invariants(rows, LYRICS)
    assert rows[3]["end"] <= 5.0
    assert rows[4]["start"] >= 29.0


def test_align_drops_lines_never_sung(tmp_path, monkeypatch):
    lyrics = "it's one two three\nfour five six seven eight\nnine ten eleven twelve"
    transcript = [{"start": 1.0, "end": 4.0,
                   "text": "it's one two three four five six seven eight"}]
    rows = run_align(tmp_path, monkeypatch, lyrics, transcript)
    assert {r["line_index"] for r in rows} == {0, 1}
    meta = json.loads((tmp_path / "w" / "align_meta.json").read_text())
    assert meta["dropped_lines"] == [2]
    assert meta["missing_spans"] == []


def test_align_flags_missing_span_for_unmatched_singing(tmp_path, monkeypatch):
    lyrics = "it's one two three\nfour five six seven eight"
    transcript = [
        {"start": 1.0, "end": 4.0, "text": "it's one two three four five six seven eight"},
        {"start": 20.0, "end": 23.0, "text": "completely unrelated ad lib bridge"},
    ]
    run_align(tmp_path, monkeypatch, lyrics, transcript)
    meta = json.loads((tmp_path / "w" / "align_meta.json").read_text())
    assert meta["dropped_lines"] == []
    assert meta["missing_spans"] == [[19.0, 24.0]]
