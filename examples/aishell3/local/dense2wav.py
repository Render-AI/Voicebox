import argparse
from pathlib import Path

import dac
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from audiotools import AudioSignal


def d2w(dense_dir):
    model_path = dac.utils.download(model_type="16khz")
    model = dac.DAC.load(model_path)
    model.to("cuda")

    dense_dir = Path(dense_dir)
    for pt in dense_dir.glob("**/*.pt"):
        print(pt)
        z = torch.load(pt)
        print(z.shape)

        if z.shape[1] != 1024:
            z = z.transpose(1, 2)

        # Decode audio signal
        y = model.decode(z)
        sf.write(
            pt.with_suffix(".wav"),
            y.detach().cpu().numpy().squeeze(0).squeeze(0),
            16000,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process wav to codes")
    parser.add_argument("dense_dir", type=Path, help="output dense feat dir")

    args = parser.parse_args()

    d2w(args.dense_dir)
