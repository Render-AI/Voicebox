from pathlib import Path
from time import time

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from einops import rearrange
from voicebox.data.audiotext_dataset import AudioTextDataset
from voicebox.lit_voicebox import VoiceboxLightningModule


def mel2wav(mel, wav_name):
    hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_discrete").cuda()

    def generate(x):
        # inference_padding = 5
        # x = F.pad(x, (inference_padding, inference_padding), "replicate").cuda()
        x = x.cuda()
        x = hifigan(x)
        return x

    if mel.shape[1] != 128:
        mel = torch.transpose(mel, 1, 2)

    # Generate
    wav = generate(mel).cpu()[0]
    torchaudio.save(wav_name, wav, 16000)


def main():
    root = Path("/data/Users/chenhaitao/Codes/leetcode/Voicebox/examples/ljspeech/")
    exp_dir = Path("lightning_logs/version_0")
    chkt_name = "epoch=60-step=49715.ckpt"
    with open(exp_dir / "config.yaml", "r") as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)

    model = VoiceboxLightningModule.load_from_checkpoint(
        checkpoint_path=exp_dir / f"checkpoints/{chkt_name}",
        config=configuration,
    )
    model.eval()
    test_dataset = AudioTextDataset(
        metadata_path=root / "data/eval.txt",
        phonesets_path=root / "data/phonesets.txt",
        melspec_dir=root / "data/melspec",
        text_path=root / "data/phone_transcripts.pt",
    )

    # print(test_dataset[0])

    d = test_dataset[0]

    normed_melspec = d["normed_melspec"].unsqueeze(0).transpose(1, 2).cuda()



    aligned_phones_ids = d["aligned_phones_ids"].unsqueeze(0).cuda()

    # print(melspec.shape)
    # print(aligned_phones_ids.shape)
    length = aligned_phones_ids.shape[1]
    mask = torch.zeros(aligned_phones_ids.shape).bool()


    mask_start = int(0.2 * length)
    mask_end = int(0.6 * length)

    mask[0, mask_start : mask_end] = True
    mask = mask.cuda()
    normed_melspec = normed_melspec * rearrange(~mask, "... -> ... 1")
    print(normed_melspec)

    st = time()
    with torch.no_grad():
        all_melspec = model.generate(
            phoneme_ids=aligned_phones_ids, cond=normed_melspec, mask=mask
        )
    print(f"{time() - st} sec used in Voicebox")
   
    new_melspec = torch.cat((normed_melspec[:, :mask_start], all_melspec[:, mask_start:mask_end], normed_melspec[:, mask_end:]), dim=1)

    new_melspec = new_melspec * (test_dataset.mel_std ** 2)+ test_dataset.mel_mean

    st = time()
    mel2wav(new_melspec, "test.wav")
    print(f"{time() - st} sec used in mel2wav")


if __name__ == "__main__":
    main()
