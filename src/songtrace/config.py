import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    whisper_url: str
    whisper_api_key: str | None = None
    whisper_model: str = "whisper-1"
    whisper_api: str = "openai"
    output_dir: Path = Path("output")
    work_dir: Path = Path("work")
    creator: str = "songtrace"
    demucs_model: str = "htdemucs_ft"
    crepe_model: str = "full"
    device: str = "cuda"
    bind_host: str = "127.0.0.1"
    port: int = 8000
    # YouTube player clients to try, in order, until one yields a download.
    # "default" means yt-dlp's own choice; see its extractor-args docs.
    ytdlp_clients: list[str] = field(default_factory=lambda: ["default", "tv"])
    ytdlp_cookies: Path | None = None
    ytdlp_impersonate: str | None = None
    ytdlp_pot_provider_url: str | None = None


_PATH_FIELDS = {"output_dir", "work_dir", "ytdlp_cookies"}


def load_config(path: Path | None = None) -> Config:
    if path is None:
        env = os.environ.get("SONGTRACE_CONFIG")
        path = Path(env) if env else Path.home() / ".config/songtrace/config.toml"
    data = tomllib.loads(Path(path).read_text())
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
    for key in _PATH_FIELDS & set(data):
        data[key] = Path(data[key])
    return Config(**data)
