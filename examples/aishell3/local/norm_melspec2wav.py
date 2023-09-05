import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms



def mel2wav(mel_dir, denorm=False):
    hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_discrete").cuda()

    mel_mean = -5.365301
    mel_std = 2.0389206

    
    def generate(x):
        # inference_padding = 5
        # x = F.pad(x, (inference_padding, inference_padding), "replicate").cuda()
        x = hifigan(x)
        return x
    

    for pt in sorted(mel_dir.glob("*.pt")):
        mel = torch.load(pt)

        if denorm:
            mel = mel * mel_std + mel_mean

        if mel.shape[1] != 128:
            mel = torch.transpose(mel, 1 ,2)


        # Generate
        wav = generate(mel).cpu()[0]
        torchaudio.save(pt.with_suffix(".wav"), wav, 16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process wav to mel")
    parser.add_argument("--denorm", action='store_true', help="output mel dir")
    parser.add_argument("mel_dir", type=Path, help="output mel dir")

    args = parser.parse_args()

    mel2wav(args.mel_dir, args.denorm)
