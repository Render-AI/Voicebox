import argparse
from pathlib import Path

import dac
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from audiotools import AudioSignal


def main(wav_dir, codes_dir, dense_dir):
    # Download a model
    model_path = dac.utils.download(model_type="16khz")
    model = dac.DAC.load(model_path)
    model.to("cuda")

    wav_dir = Path(wav_dir)
    codes_dir = Path(codes_dir)

    for spker in sorted(wav_dir.iterdir()):
        spker_name = spker.stem

        spker_codes_dir = codes_dir / spker_name
        if not spker_codes_dir.exists():
            spker_codes_dir.mkdir(parents=True)
        
        spker_dense_dir = dense_dir / spker_name
        if not spker_dense_dir.exists():
            spker_dense_dir.mkdir(parents=True)

        for wav in sorted(spker.glob("*.wav")):
            print(wav)
            # Load audio signal file
            # signal = AudioSignal(wav, sample_rate=16000)   do not auto resample as the api says?

            waveform, sample_rate = torchaudio.load(wav)
            signal = F.resample(waveform, sample_rate, 16000)
            signal = AudioSignal(signal, sample_rate=16000)

            # print(signal.sample_rate)
            # print(signal.duration)

            # Encode audio signal as one long file
            # (may run out of GPU memory on long files)
            signal.to(model.device)

            x = model.preprocess(signal.audio_data, signal.sample_rate)
            z, codes, latents, _, _ = model.encode(x)

            # print(z.shape)
            # print(codes.shape)


            torch.save(z, (spker_dense_dir / wav.name).with_suffix(".pt"))
            # torch.save(codes, (spker_codes_dir / wav.name).with_suffix(".pt"))


def test(codes_dir, dense_dir):
    model_path = dac.utils.download(model_type="16khz")
    model = dac.DAC.load(model_path)
    model.to("cuda")

    codes_dir = Path(codes_dir)
    for pt in codes_dir.glob("**/*.pt"):
        print(pt)
        codes = torch.load(pt)
        print(codes.shape)

        z = model.quantizer.from_codes(codes[:, :, :])[0]
        print(z.shape)

        # Decode audio signal
        y = model.decode(z)
        sf.write(
            "test_codes.wav", y.detach().cpu().numpy().squeeze(0).squeeze(0), 16000,
        )
        break
    
    dense_dir = Path(dense_dir)
    for pt in dense_dir.glob("**/*.pt"):
        print(pt)
        z = torch.load(pt)
        print(z.shape)

        # Decode audio signal
        y = model.decode(z)
        sf.write(
            "test_dense.wav", y.detach().cpu().numpy().squeeze(0).squeeze(0), 16000,
        )
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process wav to codes")
    parser.add_argument("wav_dir", type=Path, help="wav dir")
    parser.add_argument("codes_dir", type=Path, help="output codes dir")
    parser.add_argument("dense_dir", type=Path, help="output dense feat dir")

    args = parser.parse_args()

    main(args.wav_dir, args.codes_dir, args.dense_dir)
    test(args.codes_dir, args.dense_dir)
