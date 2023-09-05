import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms


def main(wav_dir, mel_dir):
    melspectrogram = transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=160,
        center=False,
        power=1.0,
        norm="slaney",
        onesided=True,
        n_mels=128,
        mel_scale="slaney",
    )
    if not mel_dir.exists():
        mel_dir.mkdir(parents=True)

    for f in wav_dir.glob("**/*.wav"):
        wav, sr = torchaudio.load(f)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        wav = F.pad(wav, ((1024 - 160) // 2, (1024 - 160) // 2), "reflect")

        mel = melspectrogram(wav)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        torch.save(mel, mel_dir / f.with_suffix(".pt").name)


def test(mel_dir):
    hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_discrete").cuda()
    
    mel = torch.load(list(mel_dir.glob("*.pt"))[0])

    inference_padding = 5

    def generate(x):
        x = F.pad(x, (inference_padding, inference_padding), "replicate").cuda()
        x = hifigan(x)
        return x

    # Generate
    wav = generate(mel).cpu()[0]
    torchaudio.save("test.wav", wav, 16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process wav to mel")
    parser.add_argument("wav_dir", type=Path, help="wav dir")
    parser.add_argument("mel_dir", type=Path, help="output mel dir")

    args = parser.parse_args()

    main(args.wav_dir, args.mel_dir)
    # test(args.mel_dir)
