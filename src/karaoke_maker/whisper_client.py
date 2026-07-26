from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Config
from .languages import LANGUAGES


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def transcribe(cfg: Config, audio_path: Path, language: str | None = None,
               client: httpx.Client | None = None) -> list[TranscriptSegment]:
    headers = {}
    if cfg.whisper_api_key:
        headers["Authorization"] = f"Bearer {cfg.whisper_api_key}"
    data = {"model": cfg.whisper_model, "response_format": "verbose_json"}
    if language:
        entry = LANGUAGES.get(language.lower())
        data["language"] = entry.iso if entry else language
    own = client is None
    client = client or httpx.Client(timeout=600)
    try:
        with audio_path.open("rb") as f:
            resp = client.post(
                f"{cfg.whisper_url.rstrip('/')}/audio/transcriptions",
                headers=headers, data=data,
                files={"file": (audio_path.name, f, "audio/wav")})
        resp.raise_for_status()
        return [TranscriptSegment(s["start"], s["end"], s["text"].strip())
                for s in resp.json().get("segments", [])]
    finally:
        if own:
            client.close()
