import logging
import os

import numpy as np
import torch

from collections import OrderedDict

from accelerate import Accelerator
from tqdm import tqdm
from transformers import CONFIG_MAPPING
# from transformers import MODEL_FOR_MASKED_LM_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, DataCollatorForLanguageModeling
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW

import wandb

TASK_MAPPING = {
    "mlm": {
        "model": AutoModelForMaskedLM,
        "collator": DataCollatorForLanguageModeling
    },
    "ner": {
        "model": AutoModelForTokenClassification,
        "collator": DataCollatorForTokenClassification
    }
}


class WikiModelFromConfig:

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    def __init__(self, config):
        self.config = config["pretrain" if "pretrain" in config else "finetune"]
        self.experiment_id = config["experiment"]["id"]
        self.wiki_id = config["wiki_id"]
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

        if "wandb_project" in config["experiment"]:
            wandb.init(
                project=f"{self.experiment_id}.{self.wiki_id}",
                entity=config["experiment"]["wandb_project"],
                config={
                    "learning_rate": self.config["lr"],
                    "architecture": self.config["model_type"],
                    "epochs": self.config["num_train_epochs"],
                    "batch_size": self.config["batch_size"],
                    "task": self.config["task"],
                    "mask_prob": self.config["mask_prob"],
                    "wiki_id": self.wiki_id,
                }
            )

    def train(self, dataset):
        raise NotImplementedError("Subclasses must implement this method.")

    def test(self, dataset):
        raise NotImplementedError("Subclasses must implement this method.")

    def save(self):

        path = self.config["export"]["path"]
        export_type = self.config["export"]["export_type"]

        if export_type == "hub":
            logging.info(f"Pushing model to hub: {path}/{self.experiment_id}.{self.wiki_id}")
            self.model.push_to_hub(
                f"{path}/{self.experiment_id}.{self.wiki_id}",
                use_temp_dir=True,
                repo_name=path,
                private=True
            )
        elif export_type == "local":
            logging.info(f"Saving model to: {path}/{self.experiment_id}/{self.wiki_id}")
            if not os.path.exists(f"{path}/{self.experiment_id}"):
                os.makedirs(f"{path}/{self.experiment_id}")
            self.model.save_pretrained(f"{path}/{self.experiment_id}/{self.wiki_id}")
        else:
            raise ValueError("Invalid export type.")


class WikiMLM(WikiModelFromConfig):
    def __init__(self, config, tokenizer, dataset):
        super().__init__(config)
        # self.__dict__.update(config["pretrain"])
        self.tokenizer = tokenizer
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.config["mask_prob"]
        )

        splits = dataset.keys()
        if "train" not in splits or "test" not in splits:
            raise ValueError("Both train and test splits must be present in the dataset.")
        if len(splits) > 2:
            logging.warning("More than two splits present. Ignoring all but train and test.")

        logging.info("Tokenizing and batching datasets...")
        self.loaders = {split: self._tokenize_and_collate(dataset[split]) for split in splits}
        if "train" in self.loaders:
            self.loaders["train"].shuffle = True

        if "from_config" in self.config:
            model_config = CONFIG_MAPPING[self.config["model_type"]].from_json_file(self.config["from_config"])
            model_config.vocab_size = self.tokenizer.vocab_size
            logging.info(f"Initializing model with config: {model_config}")
            self.model = AutoModelForMaskedLM.from_config(
                model_config
            )

        elif "from_pretrained" in self.config:
            logging.info(f"Loading model from hub: {self.config['from_pretrained']}.{self.wiki_id}")
            # TODO: Fix hardcoding of model type with wiki_id. This should be configurable.
            self.model = AutoModelForMaskedLM.from_pretrained(
                f"{self.config['from_pretrained']}.{self.wiki_id}"
            )

        else:
            raise ValueError("`from_config` or `from_pretrained` must be in the configuration.")

        self.model = self.model.to(self.device, dtype=self.torch_dtype)
        logging.info(f"{self.model.config.model_type} for MLM loaded.")
        logging.info(f"Number of parameters: {round(self.model.num_parameters() / 1e6)}M")

        self.accelerator = Accelerator()

    def _tokenize_and_collate(self, dataset):

        batched_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                max_length=self.config["max_length"],
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
            batch_size=self.config["batch_size"]
        )

        return loader

    def _eval_loop(self, loader):

        self.model.eval()
        losses = []
        for batch in loader:
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                losses.append(
                    self.accelerator.gather_for_metrics(loss.repeat(self.config["batch_size"]))
                )
        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)
        perplexity = torch.exp(eval_loss)

        return eval_loss, perplexity

    def train(self, dataset):

        num_train_epochs = self.config["num_train_epochs"]
        num_train_steps = num_train_epochs * len(self.loaders["train"])
        optimizer = AdamW(self.model.parameters(), lr=float(self.config["lr"]))
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )

        self.model, optimizer, self.loaders["train"], self.loaders["test"] = \
            self.accelerator.prepare(
                self.model,
                optimizer,
                self.loaders["train"],
                self.loaders["test"]
            )

        logging.info(f"Training for {num_train_epochs} epochs with {num_train_steps} steps.")
        progress_bar = tqdm(range(num_train_steps))
        for epoch in range(num_train_epochs):
            self.model.train()
            for i, batch in enumerate(self.loaders["train"]):
                outputs = self.model(**batch)
                loss = outputs.loss
                wandb.log({"train_loss": loss.item()})
                wandb.log({"train_ppl": torch.exp(loss).item()})
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                if i > 0 and i % self.config["eval_steps"] == 0:
                    eval_loss, perplexity = self._eval_loop(self.loaders["test"])
                    wandb.log({"eval_loss": eval_loss.item()})
                    wandb.log({"eval_ppl": perplexity.item()})

                if i > 0 and i % self.config["save_steps"] == 0:
                    self.save()

                self.model.train()

        self.accelerator.end_training()
        eval_loss, perplexity = self._eval_loop(self.loaders["test"])
        wandb.log({"eval_loss": eval_loss.item()})
        wandb.log({"eval_ppl": perplexity.item()})

        logging.info("Training complete.")

    def test(self, dataset):

        loader = self._tokenize_and_collate(dataset)
        loader = self.accelerator.prepare(loader)
        loss, perplexity = self._eval_loop(loader)

        wandb.summary["test_loss"] = loss.item()
        wandb.summary["test_ppl"] = perplexity.item()


class WikiNER(WikiModelFromConfig):
    def __init__(self, config, tokenizer, dataset):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.label_set = dataset["train"].features["ner_tags"].feature.names
        self.label_to_id = {label: i for i, label in enumerate(self.label_set)}
        self.id_to_label = {i: label for i, label in enumerate(self.label_set)}
        self.collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        splits = dataset.keys()
        logging.info("Tokenizing and batching datasets...")
        self.loaders = {split: self._tokenize_and_collate(dataset[split]) for split in splits}
        if "train" in self.loaders:
            self.loaders["train"].shuffle = True

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config["from_pretrained"],
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )
        self.model = self.model.to("cuda")
        self.accelerator = Accelerator()

    def _align_labels(self, example):
        tokenized_input = self.tokenizer(
            example["tokens"],
            is_split_into_words=True
        )

        seen = set()
        labels = []
        for idx in tokenized_input.word_ids()[1:-1]:
            if idx in seen:
                labels.append(-100)
            else:
                labels.append(example['ner_tags'][idx])
                seen.add(idx)

        tokenized_input["labels"] = [-100] + labels + [-100]

        return tokenized_input

    def _tokenize_and_collate(self, dataset):

        batched_dataset = dataset.map(
            self._align_labels,
            remove_columns=dataset.column_names
        )

        loader = DataLoader(
            batched_dataset,
            collate_fn=self.collator,
            batch_size=self.config["batch_size"]
        )

        return loader

    def _process_outputs_for_eval(self, outputs):

        predictions = outputs.logits.argmax(dim=-1)
        labels = outputs["labels"]

        # Necessary to pad predictions and labels for being gathered
        predictions = self.accelerator.pad_across_processes(
            predictions, dim=1, pad_index=-100
        )
        labels = self.accelerator.pad_across_processes(
            labels, dim=1, pad_index=-100
        )

        preds = self.accelerator.gather(predictions).detach().cpu().clone().numpy()
        labels = self.accelerator.gather(labels).detach().cpu().clone().numpy()

        true_labels = [l for label in labels for l in label if l != -100]
        true_preds = [
            p for prediction, label in zip(predictions, labels)
            for p, l in zip(prediction, label) if l != -100
        ]

        return true_labels, true_preds

    def train(self):

        num_train_epochs = self.config["num_train_epochs"]
        num_train_steps = num_train_epochs * len(self.loaders["train"])
        optimizer = AdamW(self.model.parameters(), lr=float(self.config["lr"]))
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )

        self.model, optimizer = self.accelerator.prepare(self.model, optimizer)
        for k, loader in self.loaders.items():
            self.loaders[k] = self.accelerator.prepare(loader)

        logging.info(f"Training for {num_train_epochs} epochs with {num_train_steps} steps.")
        progress_bar = tqdm(range(num_train_steps))
        for epoch in range(num_train_epochs):
            self.model.train()
            for i, batch in enumerate(self.loaders["train"]):
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            self.model.eval()
            running_loss = 0.0
            for dev_batch in self.loaders["dev"]:
                with torch.no_grad():
                    outputs = self.model(**dev_batch)
                true_preds, true_labels = self._process_outputs_for_eval(outputs)

            val_loss = running_loss / len(self.loaders["dev"])
            progress_bar.set_description(f"Validation loss: {val_loss}")
            self.model.train()
