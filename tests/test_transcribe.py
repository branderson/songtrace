from songtrace.lyrics import normalize_lyrics
from songtrace.whisper_client import TranscriptSegment
from songtrace.stages.transcribe import classify_lyrics_match, lyrics_similarity, vocal_sections

def seg(s, e, t=""):
    return TranscriptSegment(start=s, end=e, text=t)

def test_vocal_sections_merges_close_segments():
    sections = vocal_sections([seg(2, 5), seg(6, 9), seg(30, 33)])
    assert sections == [(1.0, 10.0), (29.0, 34.0)]

def test_vocal_sections_handles_out_of_order_segments():
    sections = vocal_sections([seg(30, 33), seg(1, 4)])
    assert sections == [(0.0, 5.0), (29.0, 34.0)]

def test_vocal_sections_keeps_contained_segment():
    sections = vocal_sections([seg(1, 10), seg(3, 5)])
    assert sections == [(0.0, 11.0)]

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
    sections = vocal_sections(segs)
    dropped, missing = classify_lyrics_match(segs, ref, sections)
    assert dropped == [2]
    assert missing == []

def test_classify_lyrics_match_flags_unmatched_singing():
    ref = normalize_lyrics("it's one two three\nfour five six seven eight")
    segs = [seg(1, 4, "it's one two three four five six seven eight"),
            seg(20, 23, "completely unrelated ad lib bridge")]
    sections = vocal_sections(segs)
    dropped, missing = classify_lyrics_match(segs, ref, sections)
    assert dropped == []
    assert missing == [(19.0, 24.0)]

def test_classify_lyrics_match_keeps_partially_matched_lines():
    ref = normalize_lyrics("it's one two three\nfour five six seven eight")
    # line 1 heard with one word garbled ("five" -> "fife"): still a replace/equal
    # match, so the line stays matched rather than dropped
    segs = [seg(1, 4, "it's one two three four fife six seven eight")]
    sections = vocal_sections(segs)
    dropped, _ = classify_lyrics_match(segs, ref, sections)
    assert dropped == []
