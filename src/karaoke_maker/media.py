import json
import subprocess
from pathlib import Path

from .pipeline import PipelineError


def ffmpeg(*args: str) -> None:
    result = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"ffmpeg failed: {result.stderr[-2000:]}")


def ffprobe_codec(path: Path) -> tuple[str, str]:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"ffprobe failed on {path}")
    data = json.loads(result.stdout)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), {})
    return data["format"]["format_name"], video.get("codec_name", "")
