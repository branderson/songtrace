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
    iso_language = None
    if language:
        entry = LANGUAGES.get(language.lower())
        iso_language = entry.iso if entry else language

    own = client is None
    client = client or httpx.Client(timeout=600)
    try:
        with audio_path.open("rb") as f:
            if cfg.whisper_api == "asr-webservice":
                # onerahmet/openai-whisper-asr-webservice: fixed /asr endpoint,
                # file field is "audio_file", and language/output are query
                # params rather than form fields - there is no "model" param,
                # the model is fixed by how that server was started.
                params = {"output": "json"}
                if iso_language:
                    params["language"] = iso_language
                resp = client.post(
                    f"{cfg.whisper_url.rstrip('/')}/asr",
                    headers=headers, params=params,
                    files={"audio_file": (audio_path.name, f, "audio/wav")})
            else:
                data = {"model": cfg.whisper_model, "response_format": "verbose_json"}
                if iso_language:
                    data["language"] = iso_language
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
