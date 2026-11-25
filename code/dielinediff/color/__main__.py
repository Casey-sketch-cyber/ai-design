"""Create a color projection LUT from a CMYK ICC profile."""

import argparse
import json
from pathlib import Path

from . import fetch_eci_profile, save_lut


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--icc", help="Path to an existing CMYK ICC profile.")
    parser.add_argument("--output", required=True, help="Output .npy file.")
    parser.add_argument("--cache-dir", help="Directory in which to cache the public ECI profile.")
    parser.add_argument("--size", type=int, default=33)
    args = parser.parse_args()
    cache_dir = args.cache_dir or Path(args.output).parent
    icc = args.icc or fetch_eci_profile(cache_dir)
    print(json.dumps(save_lut(icc, args.output, args.size), indent=2))


if __name__ == "__main__":
    main()
