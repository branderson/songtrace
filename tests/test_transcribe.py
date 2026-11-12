from songtrace.lyrics import normalize_lyrics
from songtrace.whisper_client import TranscriptSegment
from songtrace.stages.transcribe import classify_lyrics_match, lyrics_similarity

def seg(s, e, t=""):
    return TranscriptSegment(start=s, end=e, text=t)

def test_similarity_high_for_matching_lyrics():
    ref = normalize_lyrics("it's one two three\nfour five six seven eight")
    segs = [seg(0, 2, "It's one two three"), seg(2, 4, "four five six seven eight")]
    assert lyrics_similarity(segs, ref) > 0.9

def test_similarity_low_for_wrong_lyrics():
    ref = normalize_lyrics("completely different words here entirely")
    segs = [seg(0, 2, "it's one two three four five six seven eight")]
    assert lyrics_similarity(segs, ref) < 0.4

def test_classify_lyrics_match_flags_extra_reference_line():
    ref = normalize_lyrics("it's one two three\nfour five six seven eight\nnine ten eleven")
    segs = [seg(1, 4, "it's one two three four five six seven eight")]
    dropped, missing = classify_lyrics_match(segs, ref)
    assert dropped == [2]
    assert missing == []

def test_classify_lyrics_match_flags_unmatched_singing():
    ref = normalize_lyrics("it's one two three\nfour five six seven eight")
    segs = [seg(1, 4, "it's one two three four five six seven eight"),
            seg(20, 23, "completely unrelated ad lib bridge")]
    dropped, missing = classify_lyrics_match(segs, ref)
    assert dropped == []
    assert missing == [(19.0, 24.0)]

def test_classify_lyrics_match_keeps_partially_matched_lines():
    ref = normalize_lyrics("it's one two three\nfour five six seven eight")
    # line 1 heard with one word garbled ("five" -> "fife"): still a replace/equal
    # match, so the line stays matched rather than dropped
    segs = [seg(1, 4, "it's one two three four fife six seven eight")]
    dropped, _ = classify_lyrics_match(segs, ref)
    assert dropped == []

def test_classify_lyrics_match_scopes_missing_span_to_adlib_not_whole_section():
    # a matched line and an unmatched ad-lib close together: missing_spans
    # must stay tight around the ad-lib so the review marks the right place.
    ref = normalize_lyrics("it's one two three\nfour five six seven eight")
    segs = [seg(1, 4, "it's one two three four five six seven eight"),
            seg(6, 8, "completely unrelated ad lib")]
    dropped, missing = classify_lyrics_match(segs, ref)
    assert dropped == []
    assert missing == [(5.0, 9.0)]
    for a, b in missing:
        assert not (a <= 1.0 < b)        # matched line's start (1.0) untouched


THREE_LINES = "one two three\nfour five six\nseven eight nine"


def test_lines_whisper_missed_are_kept_when_singing_went_untranscribed():
    ref = normalize_lyrics(THREE_LINES)
    segs = [seg(0, 2, "one two three"), seg(20, 22, "seven eight nine")]
    assert classify_lyrics_match(segs, ref)[0] == [1]
    dropped, _ = classify_lyrics_match(segs, ref, unheard=[(8.0, 14.0)])
    assert dropped == []


def test_untranscribed_singing_elsewhere_does_not_keep_a_line():
    ref = normalize_lyrics(THREE_LINES)
    segs = [seg(0, 2, "one two three"), seg(20, 22, "seven eight nine")]
    dropped, _ = classify_lyrics_match(segs, ref, unheard=[(30.0, 35.0)])
    assert dropped == [1]


def test_singing_before_anything_heard_keeps_the_first_line():
    ref = normalize_lyrics(THREE_LINES)
    segs = [seg(10, 12, "four five six"), seg(20, 22, "seven eight nine")]
    assert classify_lyrics_match(segs, ref)[0] == [0]
    assert classify_lyrics_match(segs, ref, unheard=[(3.0, 9.0)])[0] == []


def test_a_repeated_chorus_still_matches():
    # over 200 words, with the chorus's words each over 1% of them: difflib's
    # autojunk would refuse to match any of them
    chorus = "and you should be sad\ni'm sorry you should be sad"
    verses = [f"verse {n} line {k} " + " ".join(f"w{n}{k}{j}" for j in range(6))
              for n in range(4) for k in range(4)]
    lyrics = "\n".join(sum(([v, chorus] for v in verses), []))
    ref = normalize_lyrics(lyrics)
    assert len(ref.words) > 200
    # every verse misheard, every chorus heard right
    heard = sum(([f"misheard{n} " * 8, chorus.replace("\n", " ")]
                 for n in range(len(verses))), [])
    segs = [seg(0, 1, " ".join(heard))]
    assert lyrics_similarity(segs, ref) > 0.4
