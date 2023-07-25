from pathlib import Path

import numpy as np
import torch


def main():
    root = Path("data/melspec")
    mels = []
    for i, pt in enumerate(sorted(root.glob("*.pt"))):
        print(i, pt)
        a = torch.load(pt).cpu().numpy()
        print(a.shape)
        mels.append(a)

    mel = np.concatenate(mels, axis=-1)
    print(mel.shape)

    print(mel.mean())
    print(mel.std())


if __name__ == "__main__":
    main()
