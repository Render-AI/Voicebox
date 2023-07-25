from pathlib import Path

import argparse

def main():
    parser = argparse.ArgumentParser(description="Process textgrid to lab")
    parser.add_argument("--ratio", type=str, default="0.995,0.0025,0.0025", help="train:eval:test ratio")
    parser.add_argument("wav_dir", type=Path, help="wav dir")
    parser.add_argument("output_dir", type=Path, help="output split")
    args = parser.parse_args()

    wav_list = sorted(list(args.wav_dir.glob("**/*.wav")))
    wav_list = [p.stem for p in wav_list]
    tot = len(wav_list)
    ratios = args.ratio.split(",")
    num_train = int(tot * float(ratios[0]))
    num_eval = int(tot * float(ratios[1]))
    num_test = int(tot * float(ratios[2]))

    with open(args.output_dir / "train.txt", 'w') as fout:
        fout.write("\n".join(wav_list[:num_train]))

    with open(args.output_dir / "eval.txt", 'w') as fout:
        fout.write("\n".join(wav_list[num_train:num_train+num_eval]))
    
    with open(args.output_dir / "test.txt", 'w') as fout:
        fout.write("\n".join(wav_list[num_train+num_eval:]))



if __name__ == "__main__":
    main()
