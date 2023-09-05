import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Process textgrid to lab")
    parser.add_argument("--unaligned", type=Path, default="data/TextGrid/unaligned.txt", help="unaligned wav list")
    parser.add_argument("wav_dir", type=Path, help="wav dir")
    parser.add_argument("output_dir", type=Path, help="output split")
    args = parser.parse_args()

    unaligned_wavs = []
    if args.unaligned.exists():
        with open(args.unaligned, 'r') as fin:
            for line in fin:
                wav, _ = line.split(maxsplit=1)
                unaligned_wavs.append(wav)

    train_dir = args.wav_dir / "train"
    train_wav_list = sorted(list(train_dir.glob("**/*.wav")))
    train_wav_list = [f"train/{p.stem[:7]}/{p.stem}"  for p in train_wav_list if not p.stem in unaligned_wavs]
    with open(args.output_dir / "train.txt", "w") as fout:
        fout.write("\n".join(train_wav_list[:-10]))

    with open(args.output_dir / "eval.txt", "w") as fout:
        fout.write("\n".join(train_wav_list[-10:]))

    test_dir = args.wav_dir / "test"
    test_wav_list = sorted(list(test_dir.glob("**/*.wav")))
    test_wav_list = [f"test/{p.stem[:7]}/{p.stem}"  for p in test_wav_list if not p.stem in unaligned_wavs]
    with open(args.output_dir / "test.txt", "w") as fout:
        fout.write("\n".join(test_wav_list))


if __name__ == "__main__":
    main()
