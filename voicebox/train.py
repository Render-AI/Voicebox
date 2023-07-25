import lightning as L
import torch
from lightning.pytorch.cli import LightningCLI

from voicebox.lit_datamodule import AudioTextDataModule
from voicebox.lit_voicebox import VoiceboxLightningModule


def cli_main():
    cli = LightningCLI(VoiceboxLightningModule, AudioTextDataModule)


if __name__ == "__main__":
    cli_main()
