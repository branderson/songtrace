import re
from dataclasses import dataclass
from num2words import num2words

from .languages import LANGUAGES

SECTION_RE = re.compile(r"^\s*[\[(].*[\])]\s*$")


@dataclass
class RefLyrics:
    lines: list[list[str]]
    words: list[str]
    word_line_index: list[int]


def normalize_lyrics(text: str) -> RefLyrics:
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or SECTION_RE.match(raw):
            continue
        lines.append(raw.split())
    words, idx = [], []
    for i, line in enumerate(lines):
        words.extend(line)
        idx.extend([i] * len(line))
    return RefLyrics(lines, words, idx)


def alignment_token(word: str, language: str | None = None) -> str:
    entry = LANGUAGES.get((language or "english").lower())
    lang = entry.iso if entry else "en"

    def expand(m):
        try:
            return " " + num2words(int(m[0]), lang=lang) + " "
        except NotImplementedError:
            return " " + num2words(int(m[0])) + " "

    word = word.replace("’", "'").replace("ʼ", "'")
    word = re.sub(r"\d+", expand, word)
    word = word.lower()
    word = re.sub(r"[^a-z'À-ɏЀ-ӿ ]", "", word)
    return re.sub(r"\s+", "", word)
