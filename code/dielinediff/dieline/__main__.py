"""Write the default or a custom RTE carton geometry."""

import argparse
from dataclasses import fields
from pathlib import Path

from .geometry import BoxSpec, build_geometry, write_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-side", type=int, default=1024)
    for field in fields(BoxSpec):
        parser.add_argument("--" + field.name.replace("_", "-"), type=float, default=field.default)
    args = vars(parser.parse_args())
    output = args.pop("output")
    short_side = args.pop("short_side")
    write_geometry(build_geometry(BoxSpec(**args), short_side), output)


if __name__ == "__main__":
    main()
