import json

import numpy as np
import pytest
import soundfile as sf

from songtrace import edits
from songtrace.config import Config
from songtrace.lyrics import normalize_lyrics
from songtrace.pipeline import Ctx, SongInput
from songtrace.stages import align
from songtrace.whisper_client import TranscriptSegment

REAL_UNTRANSCRIBED_SINGING = align.untranscribed_singing


@pytest.fixture(autouse=True)
def no_untranscribed_singing(monkeypatch):
    """The tests have no vocals.wav; those about missed singing set their own."""
    monkeypatch.setattr(align, "untranscribed_singing", lambda vocals, segments: [])


# the vocals the fake emission stands for: 60s of 20ms frames
DURATION = 60.0


@pytest.fixture(autouse=True)
def fake_emission(monkeypatch):
    monkeypatch.setattr(align, "vocal_emission",
                        lambda vocals, device: (np.zeros((3000, 1)), 0.02))


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


def fake_align(emission, start, end, tokens):
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


def test_a_transcript_unlike_the_lyrics_aligns_every_line_with_a_warning(
        tmp_path, monkeypatch):
    lyrics = "totally unrelated nonsense\nwords banana"
    rows = run_align(tmp_path, monkeypatch, lyrics)
    check_invariants(rows, lyrics)
    meta = json.loads((tmp_path / "w" / "align_meta.json").read_text())
    assert meta["dropped_lines"] == [] and meta["missing_spans"] == []
    assert "similarity 0.00" in meta["warnings"][0]


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
    assert rows[-1]["end"] <= DURATION


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


def _tone(seconds, sr, level):
    t = np.arange(int(seconds * sr)) / sr
    return level * np.sin(2 * np.pi * 220 * t)


def test_untranscribed_singing_finds_sung_audio_the_transcript_skips(
        tmp_path):
    sr = 16000
    rng = np.random.default_rng(0)
    audio = np.concatenate([
        _tone(2, sr, 0.3),           # 0-2s: sung and transcribed
        np.zeros(2 * sr),            # 2-4s: silence
        _tone(5, sr, 0.3),           # 4-9s: sung, never transcribed
        np.zeros(3 * sr),            # 9-12s: silence
        _tone(1, sr, 0.3),           # 12-13s: too short to count
        np.zeros(3 * sr),
    ]) + 0.001 * rng.standard_normal(16 * sr)   # separation bleed
    path = tmp_path / "vocals.wav"
    sf.write(path, audio.astype(np.float32), sr)
    segments = [TranscriptSegment(0.0, 2.0, "one two")]
    spans = REAL_UNTRANSCRIBED_SINGING(path, segments)
    assert len(spans) == 1
    assert spans[0][0] == pytest.approx(4.0, abs=0.1)
    assert spans[0][1] == pytest.approx(9.0, abs=0.1)


def test_lines_whisper_missed_are_kept_when_there_is_singing_for_them(
        tmp_path, monkeypatch):
    lyrics = "one two three\nfour five six\nseven eight nine"
    transcript = [{"start": 1.0, "end": 3.0, "text": "one two three"},
                  {"start": 40.0, "end": 42.0, "text": "seven eight nine"}]
    monkeypatch.setattr(align, "untranscribed_singing",
                        lambda vocals, segments: [(20.0, 26.0)])
    rows = run_align(tmp_path, monkeypatch, lyrics, transcript)
    check_invariants(rows, lyrics)
    meta = json.loads((tmp_path / "w" / "align_meta.json").read_text())
    assert meta["dropped_lines"] == []


def test_stars_bracket_each_line_and_take_no_words(tmp_path, monkeypatch):
    lyrics = "one two\n...\nthree four"
    transcript = [{"start": 0.0, "end": 10.0, "text": "one two three four"}]
    seen = []

    def spy(emission, start, end, tokens):
        seen.append(tokens)
        # each star claims a long stretch, so a word given a star's span shows
        spans = [(start + i, start + i + (5.0 if t == "*" else 0.5), 0.9)
                 for i, t in enumerate(tokens)]
        return spans

    monkeypatch.setattr(align, "force_align_section", spy)
    ctx = make_ctx(tmp_path, lyrics, transcript)
    align.align_stage().run(ctx)
    rows = json.loads((ctx.work / "alignment.json").read_text())
    assert seen == [["*", "one", "two", "*", "three", "four", "*"]]
    timed = [r for r in rows if r["word"] != "..."]
    assert all(r["end"] - r["start"] == pytest.approx(0.5) for r in timed)
    check_invariants(rows, lyrics)


def test_the_whole_song_is_aligned_at_once(tmp_path, monkeypatch):
    calls = []

    def spy(emission, start, end, tokens):
        calls.append((start, end))
        return fake_align(emission, start, end, tokens)

    transcript = [{"start": 1.0, "end": 4.0, "text": "it's one two three"},
                  {"start": 30.0, "end": 33.0, "text": "four five six seven eight"}]
    monkeypatch.setattr(align, "force_align_section", spy)
    align.align_stage().run(make_ctx(tmp_path, transcript=transcript))
    assert calls == [(0.0, DURATION)]


def pin(ctx, lyrics, anchors):
    data = edits.load(ctx.work, normalize_lyrics(lyrics).lines)
    edits.set_anchors(data, anchors)
    edits.save(ctx.work, data)


def test_pinned_lines_split_the_song_and_start_where_pinned(tmp_path, monkeypatch):
    lyrics = "one two\nthree four\nfive six"
    transcript = [{"start": 0.0, "end": 10.0, "text": "one two three four five six"}]
    calls = []

    def spy(emission, start, end, tokens):
        calls.append((start, end, tokens))
        return fake_align(emission, start, end, tokens)

    monkeypatch.setattr(align, "force_align_section", spy)
    ctx = make_ctx(tmp_path, lyrics, transcript)
    pin(ctx, lyrics, {"1": 20.0})
    align.align_stage().run(ctx)
    slack = align.PIN_SLACK_SEC
    assert calls == [
        (0.0, 20.0 - slack, ["*", "one", "two", "*"]),
        # no star before a pinned line: nothing may stand in for its start
        (20.0 - slack, DURATION, ["three", "four", "*", "five", "six", "*"]),
    ]
    rows = json.loads((ctx.work / "alignment.json").read_text())
    check_invariants(rows, lyrics)
    assert rows[2]["start"] == pytest.approx(20.0 - slack)
    meta = json.loads((ctx.work / "align_meta.json").read_text())
    assert meta["anchors"] == {"1": 20.0}


def test_a_pin_on_the_first_line_drops_the_leading_star(tmp_path, monkeypatch):
    seen = []

    def spy(emission, start, end, tokens):
        seen.append((start, tokens[0]))
        return fake_align(emission, start, end, tokens)

    monkeypatch.setattr(align, "force_align_section", spy)
    ctx = make_ctx(tmp_path)
    pin(ctx, LYRICS, {"0": 12.0})
    align.align_stage().run(ctx)
    assert seen == [(12.0 - align.PIN_SLACK_SEC, "it's")]


def test_a_pinned_line_is_never_dropped(tmp_path, monkeypatch):
    lyrics = "it's one two three\nfour five six seven eight\nnine ten eleven twelve"
    transcript = [{"start": 1.0, "end": 4.0,
                   "text": "it's one two three four five six seven eight"}]
    monkeypatch.setattr(align, "force_align_section", fake_align)
    ctx = make_ctx(tmp_path, lyrics, transcript)
    pin(ctx, lyrics, {"2": 40.0})
    align.align_stage().run(ctx)
    rows = json.loads((ctx.work / "alignment.json").read_text())
    check_invariants(rows, lyrics)
    meta = json.loads((ctx.work / "align_meta.json").read_text())
    assert meta["dropped_lines"] == []


def test_force_align_section_gives_up_on_a_window_too_short(monkeypatch):
    pytest.importorskip("torchaudio")
    frames = np.zeros((5, 29), dtype=np.float32)
    assert align.force_align_section((frames, 0.02), 0.0, 0.1, ["hello"]) == []


def test_only_the_pinned_line_loses_its_star(tmp_path, monkeypatch):
    lyrics = "one\n...\ntwo three"
    transcript = [{"start": 0.0, "end": 10.0, "text": "one two three"}]
    seen = []

    def spy(emission, start, end, tokens):
        seen.append(tokens)
        return fake_align(emission, start, end, tokens)

    monkeypatch.setattr(align, "force_align_section", spy)
    ctx = make_ctx(tmp_path, lyrics, transcript)
    pin(ctx, lyrics, {"1": 20.0})
    align.align_stage().run(ctx)
    assert seen == [["*", "one", "*"], ["*", "two", "three", "*"]]
