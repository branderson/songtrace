import json
import shutil
from pathlib import Path

from ..config import Config
from ..media import ffmpeg
from ..pipeline import Ctx, PipelineError, Stage

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif"}
# yt-dlp's in-progress scratch files, which a failed attempt leaves behind.
PARTIAL_EXTENSIONS = {".part", ".ytdl", ".temp"}


def _classify_downloads(dest: Path) -> tuple[Path, Path | None]:
    files = [p for p in dest.glob("source_video.*")
             if p.suffix.lower() not in PARTIAL_EXTENSIONS]
    thumbs = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    videos = [p for p in files if p.suffix.lower() not in IMAGE_EXTENSIONS]
    if not videos:
        raise PipelineError("yt-dlp produced no video file")
    video = max(videos, key=lambda p: p.stat().st_size)
    return video, thumbs[0] if thumbs else None


class _YdlLog:
    """Routes yt-dlp's own messages into the job log, minus progress noise."""

    def __init__(self, log):
        self._log = log

    def debug(self, msg: str) -> None:
        if not msg.startswith("[debug] "):
            self._log(msg)

    def warning(self, msg: str) -> None:
        self._log(msg)

    error = warning


def _ydl_opts(cfg: Config, dest: Path, client: str, log) -> dict:
    extractor_args: dict[str, dict[str, list[str]]] = {}
    if client != "default":
        extractor_args["youtube"] = {"player_client": [client]}
    if cfg.ytdlp_pot_provider_url:
        extractor_args["youtubepot-bgutilhttp"] = {
            "base_url": [cfg.ytdlp_pot_provider_url]}
    opts = {
        "outtmpl": str(dest / "source_video.%(ext)s"),
        "format": "bestvideo*+bestaudio/best",
        "writethumbnail": True,
        "noplaylist": True,
        "noprogress": True,
        "logger": _YdlLog(log),
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 5,
        "extractor_args": extractor_args,
    }
    if cfg.ytdlp_cookies:
        opts["cookiefile"] = str(cfg.ytdlp_cookies)
    if cfg.ytdlp_impersonate:
        from yt_dlp.networking.impersonate import ImpersonateTarget
        opts["impersonate"] = ImpersonateTarget.from_str(cfg.ytdlp_impersonate)
    return opts


def download_youtube(url: str, dest: Path, cfg: Config,
                     log=print) -> tuple[Path, Path | None]:
    import yt_dlp
    clients = list(cfg.ytdlp_clients) or ["default"]
    last: Exception | None = None
    for client in clients:
        try:
            with yt_dlp.YoutubeDL(_ydl_opts(cfg, dest, client, log)) as ydl:
                ydl.extract_info(url, download=True)
            return _classify_downloads(dest)
        except Exception as e:
            last = e
            log(f"yt-dlp player_client={client} failed: {e}")
    raise PipelineError(
        f"yt-dlp {yt_dlp.version.__version__} could not download {url} after "
        f"trying player_client={', '.join(clients)}. Last error: {last}. "
        "YouTube blocks change often: upgrade yt-dlp, then see the yt-dlp "
        "options in config.toml (cookies, impersonation, PO token provider).")


def _run(ctx: Ctx) -> None:
    if bool(ctx.inp.youtube_url) == bool(ctx.inp.video_path):
        raise PipelineError("provide exactly one of youtube_url / video_path")
    thumb = None
    if ctx.inp.youtube_url:
        video, thumb = download_youtube(ctx.inp.youtube_url, ctx.work,
                                       ctx.cfg, ctx.log)
    else:
        video = ctx.work / f"source_video{ctx.inp.video_path.suffix}"
        shutil.copyfile(ctx.inp.video_path, video)
    ffmpeg("-i", str(video), "-vn", "-ac", "2", "-ar", "44100",
           str(ctx.work / "audio.wav"))
    meta = {"video_file": video.name, "thumbnail": thumb.name if thumb else None}
    (ctx.work / "meta.json").write_text(json.dumps(meta))


def acquire_stage() -> Stage:
    return Stage(
        name="acquire",
        outputs=lambda ctx: [ctx.work / "meta.json", ctx.work / "audio.wav"],
        run=_run,
    )
