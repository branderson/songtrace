from karaoke_maker.lyrics import normalize_lyrics
from karaoke_maker.whisper_client import TranscriptSegment
from karaoke_maker.stages.transcribe import lyrics_similarity, vocal_sections

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
