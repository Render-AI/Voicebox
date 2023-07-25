import argparse
import re
from pathlib import Path

from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser(description="Process lexicon")
    parser.add_argument("original_lexicon", type=Path, help="original lexicon path")
    parser.add_argument("processed_lexicon", type=Path, help="processed lexicon path")
    args = parser.parse_args()

    sp = re.compile("\s+")
    with open(args.original_lexicon) as lexicon, open(args.processed_lexicon, "w") as f:
        for line in lexicon:
            word, *phonemes = sp.split(line.strip())
            phonemes = " ".join(phonemes)
            f.write(f"{word}\t{phonemes}\n")


if __name__ == "__main__":
    main()
