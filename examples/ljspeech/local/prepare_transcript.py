import argparse
import re
from pathlib import Path

from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate transcript")
    parser.add_argument("meta", type=Path, help="LJSpeech meta path")
    parser.add_argument("output_dir", type=Path, help="output dir")
    args = parser.parse_args()

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    lines = open(args.meta, "r").readlines()
    for line in tqdm(lines):
        fn, _, transcript = line.strip().split("|")
        ident = fn
        open(args.output_dir / f"{ident}.txt", "w").write(transcript)


if __name__ == "__main__":
    main()
