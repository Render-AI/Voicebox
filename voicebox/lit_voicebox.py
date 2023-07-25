from pathlib import Path
from typing import Dict

import lightning as L
import torch
import torchaudio
from torch.optim import AdamW
from voicebox.model import ConditionalFlowMatcherWrapper, VoiceBox
from voicebox.model.lr_schedulers import WarmupCosineLRSchedule


class VoiceboxLightningModule(L.LightningModule):
    def __init__(
        self,
        model_conf={
            "dim_in": 128,
            "dim": 1024,
            "num_phoneme_tokens": 256,
            "depth": 2,
            "dim_head": 64,
            "heads": 16,
            "attention_dropout": 0.0,
            "activation_dropout": 0.1,
            "conv_pos_embed_kernel_size": 31,
            "conv_pos_embed_groups": 16,
            "conv_pos_embed_depth": 2,
            "p_drop_prob": 0.3,
        },
        opt_conf={
            "lr": 0.0002,
            "lr_init": 0.000001,
            "lr_end": 0.00001,
            "warmup_steps": 2000,
            "decay_steps": 40000,
        },
    ):
        super().__init__()

        self.opt_conf = opt_conf

        self.model = VoiceBox(**model_conf)
        self.cfm_wrapper = ConditionalFlowMatcherWrapper(
            voicebox=self.model,
            sigma=0.00001,
            cond_drop_prob=0.2,
            use_torchode=False,  # by default will use torchdiffeq with midpoint as in paper, but can use the promising torchode package too
        )

        self.save_hyperparameters()

    def training_step(self, batch: Dict, batch_idx: int):
        scheduler = self.lr_schedulers()
        # print(batch["lengths"])
        loss = self.cfm_wrapper(
            batch["melspec"],
            phoneme_ids=batch["aligned_phones_ids"],
            cond=batch["melspec"],
            seq_lens=batch["lengths"],
        )

        self.log(
            "loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "lr",
            scheduler.get_last_lr()[0],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        mask = torch.ones(
            batch["aligned_phones_ids"].shape,
            dtype=torch.long,
            device=batch["aligned_phones_ids"].device,
        ).bool()
        mel = self.cfm_wrapper.sample(
            cond=batch["melspec"], phoneme_ids=batch["aligned_phones_ids"], mask=mask
        )
        eval_dir = Path(f"{self.logger.log_dir}/eval")
        if not eval_dir.exists():
            eval_dir.mkdir(parents=True)
        torch.save(mel, f"{eval_dir}/val_{self.current_epoch}_{batch_idx}.pt")

    def generate(self, phoneme_ids, cond, mask):
        mel = self.cfm_wrapper.sample(
            phoneme_ids=phoneme_ids, cond=cond, mask=mask, steps=10, cond_scale=0.7
        )
        return mel

    def configure_optimizers(self, lr=0.0002):
        lm_opt = AdamW(
            self.model.parameters(),
            self.opt_conf["lr"],
            betas=(0.8, 0.99),
            eps=0.000000001,
            # weight_decay=4.5e-2
        )

        return {
            "optimizer": lm_opt,
            "lr_scheduler": {
                "scheduler": WarmupCosineLRSchedule(
                    lm_opt,
                    init_lr=self.opt_conf["lr_init"],
                    peak_lr=self.opt_conf["lr"],
                    end_lr=self.opt_conf["lr_end"],
                    warmup_steps=self.opt_conf["warmup_steps"],
                    total_steps=self.opt_conf["decay_steps"],
                ),
                "interval": "step",
            },
        }
