import argparse
import math
import sys
from importlib.metadata import version
from pathlib import Path

from .config import load_config
from .pipeline import STAGE_ORDER, PipelineError, SongInput, run_pipeline


def _positive_float(value: str) -> float:
    bpm = float(value)
    if not math.isfinite(bpm) or bpm <= 0:
        raise argparse.ArgumentTypeError(f"--bpm must be a finite value > 0, got {value!r}")
    return bpm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="karaoke-maker")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")

    make = sub.add_parser("make", help="build a karaoke song folder")
    make.add_argument("--title", required=True)
    make.add_argument("--artist", required=True)
    make.add_argument("--lyrics", required=True, type=Path)
    make.add_argument("--youtube")
    make.add_argument("--video", type=Path)
    make.add_argument("--genre")
    make.add_argument("--year", type=int)
    make.add_argument("--language")
    make.add_argument("--cover", type=Path)
    make.add_argument("--bpm", type=_positive_float)
    make.add_argument("--force-from", choices=STAGE_ORDER)
    make.add_argument("--config", type=Path)

    serve = sub.add_parser("serve", help="run the web UI")
    serve.add_argument("--config", type=Path)
    return parser


def _cmd_make(args) -> int:
    if bool(args.youtube) == bool(args.video):
        print("error: provide exactly one of --youtube / --video", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    inp = SongInput(
        title=args.title, artist=args.artist,
        lyrics_text=args.lyrics.read_text(encoding="utf-8"),
        youtube_url=args.youtube, video_path=args.video,
        genre=args.genre, year=args.year, language=args.language,
        cover_path=args.cover, bpm_override=args.bpm)
    try:
        result = run_pipeline(inp, cfg, force_from=args.force_from)
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"done: {result.output_dir}")
    for w in result.warnings:
        print(f"  review: {w}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(f"karaoke-maker {version('karaoke-maker')}")
        return 0
    if args.command == "make":
        return _cmd_make(args)
    if args.command == "serve":
        from .web import serve
        return serve(load_config(args.config))
    build_parser().print_help()
    return 0
