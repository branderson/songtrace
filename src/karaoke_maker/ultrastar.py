import re
from dataclasses import dataclass

NOTE_RE = re.compile(r"^([:*FRG]) (-?\d+) (\d+) (-?\d+) (.*)$")


@dataclass
class Note:
    kind: str
    beat: int
    length: int
    pitch: int
    text: str


@dataclass
class LineBreak:
    beat: int


@dataclass
class UltraStarSong:
    headers: list[tuple[str, str]]
    body: list[Note | LineBreak]


def parse_ultrastar(text: str) -> UltraStarSong:
    headers, body = [], []
    for line in text.splitlines():
        line = line.rstrip("\r")
        if not line.strip():
            continue
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            headers.append((key, value))
        elif line.startswith("E"):
            break
        elif line.startswith("-"):
            nums = line[1:].split()
            body.append(LineBreak(beat=int(nums[0])))
        else:
            m = NOTE_RE.match(line)
            if m:
                body.append(Note(m[1], int(m[2]), int(m[3]), int(m[4]), m[5]))
    return UltraStarSong(headers, body)


def serialize_ultrastar(song: UltraStarSong) -> str:
    out = [f"#{k}:{v}" for k, v in song.headers]
    for item in song.body:
        if isinstance(item, LineBreak):
            out.append(f"- {item.beat}")
        else:
            out.append(f"{item.kind} {item.beat} {item.length} {item.pitch} {item.text}")
    out.append("E")
    return "\n".join(out) + "\n"


def get_header(song: UltraStarSong, key: str) -> str | None:
    for k, v in song.headers:
        if k.upper() == key.upper():
            return v
    return None


def format_bpm(bpm: float) -> str:
    if bpm == int(bpm):
        return str(int(bpm))
    return f"{bpm:.2f}".replace(".", ",")


def parse_bpm(value: str) -> float:
    return float(value.replace(",", "."))
