import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Process textgrid to lab")
    parser.add_argument("wav_dir", type=Path, help="wav dir")
    parser.add_argument("output_dir", type=Path, help="output split")
    args = parser.parse_args()

    test_dir = args.wav_dir / "wavs"
    test_wav_list = sorted(list(test_dir.glob("**/*.wav")))
    test_wav_list = [f"wavs/{p.stem}"  for p in test_wav_list]
    with open(args.output_dir / "test.txt", "w") as fout:
        fout.write("\n".join(test_wav_list))


if __name__ == "__main__":
    main()
