import argparse
import re
from pathlib import Path

from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser(description="Process lexicon")
    parser.add_argument("phoneset", type=Path, help="original lexicon path")
    parser.add_argument("lexicon_path", type=Path, help="processed lexicon path")
    args = parser.parse_args()

    pinyins = set()
    with open(args.phoneset, "r") as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue
            elif line[0] == "#":
                continue
            else:
                phone, _ = line.split()
                pinyins.add(phone)
    
    pinyins = sorted(list(pinyins))
   
    # TODO: yu, qu, lu ... 
    with open(args.lexicon_path, "w") as fout:
        for pinyin in pinyins:
            if pinyin.startswith("a"):
                fout.write(f"{pinyin} {pinyin}")
                fout.write("\n")
            else:
                if pinyin[:-1] in ["ci", "si", "zi"]:
                    fout.write(f"{pinyin} {pinyin[:1]} ii{pinyin[-1]}")
                    fout.write("\n")
                elif pinyin[:-1] in ["chi", "shi", "zhi"]:
                    fout.write(f"{pinyin} {pinyin[:2]} iii{pinyin[-1]}")
                    fout.write("\n")
                elif pinyin[:2] in ["zh", "sh", "ch"]:
                    fout.write(f"{pinyin} {pinyin[:2]} {pinyin[2:]}")
                    fout.write("\n")
                else:
                    fout.write(f"{pinyin} {pinyin[0]} {pinyin[1:]}")
                    fout.write("\n")


if __name__ == "__main__":
    main()
