import logging

import numpy as np
import torch
import wandb

from tqdm import tqdm
from transformers import CONFIG_MAPPING
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW

from .model import ModelFromConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLM(ModelFromConfig):
    def __init__(self,
                 config,
                 tokenizer,
                 load_method,
                 load_path,
                 **kwargs
                 ):

        super().__init__(config, **kwargs)
        self.tokenizer = tokenizer
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mask_prob
        )

        if load_method == "config":
            model_config = CONFIG_MAPPING[self.model_type].from_json_file(load_path)
            model_config.vocab_size = self.tokenizer.vocab_size
            logger.info(f"Initializing model with config: {model_config}")
            self.model = AutoModelForMaskedLM.from_config(model_config)

        elif load_method == "hub":
            logger.info(f"Loading model from hub: {load_path}.")
            self.model = AutoModelForMaskedLM.from_pretrained(f"{load_path}")

        else:
            raise ValueError("`from_config` or `from_pretrained` must be in the configuration.")

        self.model = self.model.to(self.device, dtype=self.torch_dtype)
        logger.info(f"{self.model.config.model_type} for MLM loaded.")
        logger.info(f"Number of parameters: {round(self.model.num_parameters() / 1e6)}M")

    def _tokenize_and_collate(self, dataset):

        batched_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True),
            batched=True,
            remove_columns=dataset.column_names
        )

        batched_dataset = batched_dataset.remove_columns("overflow_to_sample_mapping")

        loader = DataLoader(
            batched_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=True
        )

        return loader

    def _eval_loop(self, loader):

        running = 0.0
        for batch in loader:
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                running += loss.item()
        eval_loss = running / len(loader)
        perplexity = np.exp(eval_loss)

        return eval_loss, perplexity

    def train(self, dataset):

        splits = dataset.keys()
        if "train" not in splits or "test" not in splits:
            raise ValueError("Both train and test splits must be present in the dataset.")
        if len(splits) > 2:
            logger.warning("More than two splits present. Ignoring all but train and test.")

        logger.info("Tokenizing and batching datasets...")
        loaders = {split: self._tokenize_and_collate(dataset[split]) for split in splits}
        if "train" in loaders:
            loaders["train"].shuffle = True

        num_train_epochs = self.num_train_epochs
        num_train_steps = num_train_epochs * len(loaders["train"])
        optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.lr),
            weight_decay=0.05
        )
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=num_train_steps
        )

        self.model, optimizer, loaders["train"], loaders["test"] = \
            self.accelerator.prepare(
                self.model,
                optimizer,
                loaders["train"],
                loaders["test"]
            )
        self.accelerator.register_for_checkpointing(scheduler)
        self.model.save_pretrained(self.export_path)

        logger.info(f"Training for {num_train_epochs} epochs with {num_train_steps} steps.")
        progress_bar = tqdm(range(num_train_steps))
        running_loss = np.Inf

        self.model.gradient_checkpointing_enable()

        for epoch in range(num_train_epochs):
            self.model.train()
            for i, batch in enumerate(loaders["train"]):
                outputs = self.model(**batch)
                loss = outputs.loss
                if self.wandb:
                    wandb.log({"train_loss": loss.item()})
                    wandb.log({"train_ppl": torch.exp(loss).item()})
                self.accelerator.backward(loss / 4)
                if (i + 1) % 4 == 0:
                    # loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                progress_bar.update(1)

                if i % self.eval_steps == 0:
                    self.model.eval()
                    eval_loss, perplexity = self._eval_loop(loaders["test"])
                    if self.wandb:
                        wandb.log({"eval_loss": eval_loss})
                        wandb.log({"eval_ppl": perplexity})
                    self.model.train()

            self.model.eval()
            eval_loss, perplexity = self._eval_loop(loaders["test"])
            # if self.wandb:
            #     wandb.log({"eval_loss": eval_loss})
            #     wandb.log({"eval_ppl": perplexity})
            # logger.info(f"Eval loss: {round(eval_loss, 2)}, PPL: {round(perplexity, 2)}")
            if eval_loss < running_loss:
                logger.info(f"Saving model checkpoint at epoch {epoch}.")
                self.model.save_pretrained(self.export_path)
                running_loss = eval_loss

        self.accelerator.end_training()
        eval_loss, perplexity = self._eval_loop(loaders["test"])
        if self.wandb:
            wandb.log({"eval_loss": eval_loss})
            wandb.log({"eval_ppl": perplexity})

        logger.info("Training complete.")

    def test(self, dataset):

        loader = self._tokenize_and_collate(dataset)
        loader = self.accelerator.prepare(loader) 
        loss, perplexity = self._eval_loop(loader)

        if self.wandb:
            wandb.summary["test_loss"] = round(loss, 2)
            wandb.summary["test_ppl"] = round(perplexity, 2)