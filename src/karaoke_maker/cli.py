import argparse
from importlib.metadata import version

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="karaoke-maker")
    parser.add_argument("--version", action="store_true")
    parser.add_subparsers(dest="command")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(f"karaoke-maker {version('karaoke-maker')}")
        return 0
    build_parser().print_help()
    return 0
