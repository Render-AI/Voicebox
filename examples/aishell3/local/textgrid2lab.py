import argparse
from collections import namedtuple
from pathlib import Path

import tgt
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Process textgrid to lab")
    parser.add_argument("--store_phonesets", action="store_true", help="textgrid dir")
    parser.add_argument("tg_dir", type=Path, help="textgrid dir")
    parser.add_argument("output_dir", type=Path, help="output lab dir")

    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    labs = {}
    phones = set()


    for tg in args.tg_dir.glob("**/*.TextGrid"):
        wav_stem = tg.stem
        
        f = tgt.io.read_textgrid(tg, include_empty_intervals=True)
        tier = f.get_tier_by_name("phones")
        
        lab = []
        for interval in tier.intervals:
            if wav_stem == 'LJ048-0142':
                print(interval.text)
            if interval.text == "":
                interval.text = "SIL"
            t = (interval.text, [interval.start_time, interval.end_time])
            phones.add(interval.text)
            lab.append(t)
        labs[wav_stem] = lab

    torch.save(labs, output_dir / "phone_transcripts.pt")

    if args.store_phonesets:
        with open(output_dir / "phonesets.txt", 'w') as fout:
            fout.write("\n".join(sorted(list(phones))))



if __name__ == "__main__":
    main()
