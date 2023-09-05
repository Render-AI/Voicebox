import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Process lexicon")
    parser.add_argument("script", type=Path, help="processed lexicon path")
    parser.add_argument("--output_dir", type=Path, help="processed lexicon path")
    args = parser.parse_args()

    with open(args.script) as fin:
        for line in fin:
            # print(line)
            info = line.strip().split()
            utt = info[0]
            pinyins = info[2::2]
            # print(utt, pinyins)
            with open(args.output_dir / utt[:7] / (utt[:-3] + "txt"), "w") as fout:
                fout.write(" ".join(pinyins) + "\n")


if __name__ == "__main__":
    main()
