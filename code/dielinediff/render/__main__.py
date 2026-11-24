"""Render a carton texture or generate the documented RTE self-check."""

import argparse
from pathlib import Path

from .orthographic import VIEWS, render_views
from .selfcheck import write_selfcheck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("selfcheck")
    check.add_argument("--report", type=Path)
    render = commands.add_parser("render")
    render.add_argument("--texture", type=Path, required=True)
    render.add_argument("--size", type=int, default=480)
    render.add_argument("--views", nargs="+", choices=tuple(VIEWS), default=tuple(VIEWS))
    for command in (check, render):
        command.add_argument("--geometry", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "selfcheck":
        write_selfcheck(args.geometry, args.output, args.report)
    else:
        render_views(args.geometry, args.texture, args.output, args.size, args.views)


if __name__ == "__main__":
    main()
