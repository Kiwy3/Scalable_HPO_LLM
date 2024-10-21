import sys
import json
import torch
import litgpt
from litgpt.lora import GPT, merge_lora_weights
from litgpt.data import Alpaca2k
import lightning as L
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
import numpy as np

import json
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
from lightning import LightningDataModule
from litgpt import Tokenizer
from litgpt.data import Alpaca2k, SFTDataset
from litgpt.prompts import PromptStyle
from datasets import load_dataset

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class LitLLM(L.LightningModule):
    def __init__(self, low_rank=4, rate=0.002, l_alpha=16, l_dropout=0.05,bar = True):
        super().__init__()
        # Parameters
        self.lr = rate
        self.bar = bar
        self.validation_step_outputs = []
        # Lora Model
        self.model = GPT.from_name(
            name="tiny-llama-1.1b",
            lora_r=low_rank,
            lora_alpha=l_alpha,
            lora_dropout=l_dropout,
            lora_query=True,
            lora_key=False,
            lora_value=True
        )
        litgpt.lora.mark_only_lora_as_trainable(self.model)
    
    def compute_loss(self, input_ids, targets):
        if torch.isnan(input_ids).any():
            print("NaN detected in input")

        if torch.isnan(targets).any():
            print("NaN detected in targets")
        
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        
        if torch.isnan(loss).any():
            print("NaN detected in loss")

        return loss


    #------------------------------ Training ------------------------------
    def on_train_start(self):
        state_dict = torch.load(f"checkpoints/{model_id}/lit_model.pth", mmap=True, weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)

    def training_step(self, batch):
        input_ids, targets = batch["input_ids"], batch["labels"]
        loss = self.compute_loss(input_ids, targets)
        self.log("train_loss", loss, prog_bar=self.bar)
        return loss
    
    def on_train_epoch_end(self):
        print("on_train_epoch_end")
        pass  # Disable manual checkpoint saving


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / 10)
        return [optimizer], [scheduler]
    
    #------------------------------ Validate ------------------------------
    def validation_step(self, batch):
        input_ids, targets = batch["input_ids"], batch["labels"]
        loss = self.compute_loss(input_ids, targets)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, prog_bar=self.bar)
        return loss
    
    def on_validation_epoch_end(self):
        loss_total = torch.stack(self.validation_step_outputs)
        self.log("val_loss_avg", loss_total.mean())
        return super().on_validation_epoch_end()


@dataclass
class LLMDataModule(LightningDataModule):
    mask_prompt: bool = False
    val_split_fraction: float = 0.05
    prompt_style: Union[str, PromptStyle] = "alpaca"
    ignore_index: int = -100
    seed: int = 42
    num_workers: int = 4
    download_dir: Path = Path("./data/alpaca2k")
    repo_id: str = field(repr=False, default="mhenrichsen/alpaca_2k_test")
    file_name: str = field(repr=False, default="alpaca2k_data_cleaned_archive.json")

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    val_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        # Download the dataset from Hugging Face
        load_dataset(self.repo_id, cache_dir=self.download_dir)

    def setup(self, stage: str = None) -> None:
        # Load the dataset
        dataset = load_dataset(self.repo_id, cache_dir=self.download_dir)

        # Split the dataset into training and validation sets
        train_validation_split = dataset["train"].train_test_split(test_size=self.val_split_fraction, seed=self.seed)
        train_data = train_validation_split["train"]
        val_data = train_validation_split["test"]

        # Create SFTDataset instances for training and validation
        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.val_dataset = SFTDataset(
            data=val_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        out = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed)
        )
        return out

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def BB_eval(HP):

    # Hyper Parameters loading
    grad_batches = HP.get("grad_batches", 16)
    rate = HP.get("learning_rate", 0.002)
    low_rank = HP.get("lora_rank", 4)

    # Data module management
    data_module = LLMDataModule(
        val_split_fraction=0.2,  # Adjust as needed
    )
    data_module.connect(
        tokenizer=litgpt.Tokenizer(f"checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        batch_size=1,
        max_seq_length=512
    )
    data_module.prepare_data()
    data_module.setup()

    # Configure Trainer
    trainer = L.Trainer(
            devices=1,
            max_epochs=1,
            max_steps=20,
            accumulate_grad_batches=grad_batches,
            precision="32-true",
            enable_checkpointing=False,
        )
    

    # Generate and train the model
    
    model = LitLLM(low_rank=low_rank, rate=rate)
    trainer.fit(model, datamodule = data_module)

    # Merge and compute validation loss
    merge_lora_weights(model.model)
    out = trainer.validate(model, datamodule = data_module)
    print(sum(model.validation_step_outputs))
    print(sum(model.validation_step_outputs)/200)

    validation_loss = out[0]["val_loss_avg"]

    return validation_loss


if __name__ == "__main__":
    # Hyper Parameters
    HP = {
        "learning_rate": 0.002,
        "lora_rank": 4,
    }

    out = BB_eval(HP)
    print("final output : ",out)
