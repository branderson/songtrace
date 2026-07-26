from karaoke_maker.lyrics import normalize_lyrics, alignment_token

RAW = """[Verse 1]
It's one two three

[Chorus]
Whoa, la la o' la
Count me up to 7 times
"""


def test_normalize_strips_sections_and_blanks():
    ref = normalize_lyrics(RAW)
    assert ref.lines == [
        ["It's", "one", "two", "three"],
        ["Whoa,", "la", "la", "o'", "la"],
        ["Count", "me", "up", "to", "7", "times"],
    ]
    assert ref.words[0] == "It's"
    assert ref.word_line_index == [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]


def test_alignment_token():
    assert alignment_token("Whoa,") == "whoa"
    assert alignment_token("o'") == "o'"
    assert alignment_token("7") == "seven"
    assert alignment_token("...") == ""
