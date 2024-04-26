import logging

# import numpy as np
import torch
import wandb

from accelerate import Accelerator
from tqdm import tqdm
from transformers import CONFIG_MAPPING
# from transformers import MODEL_FOR_MASKED_LM_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, DataCollatorForLanguageModeling
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW

from .model import WikiModelFromConfig
from wqe.utils.maps import TASK_TO_MODEL_AND_COLLATOR_MAPPING


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenericModelforFineTuning(WikiModelFromConfig):

    def __init__(self,
                 config,
                 tokenizer,
                 load_path,
                 label_set,
                 **kwargs
                 ):

        super().__init__(config, tokenizer, **kwargs)
        self.label_set = label_set
        self.label_to_id = {label: i for i, label in enumerate(self.label_set)}
        self.id_to_label = {i: label for i, label in enumerate(self.label_set)}
        self.collator = TASK_TO_MODEL_AND_COLLATOR_MAPPING[self.task]["collator"](tokenizer=self.tokenizer)

        self.model = TASK_TO_MODEL_AND_COLLATOR_MAPPING[self.task]["model"].from_pretrained(
            load_path,
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )
        self.model = self.model.to(self.device, dtype=self.torch_dtype)

    def _tokenize_and_collate(self, dataset):

        batched_dataset = dataset.map(
            self.tokenizer,
            remove_columns=dataset.column_names
        )

        loader = DataLoader(
            batched_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size
        )

        return loader

    def train(self, dataset):

        splits = dataset.keys()
        logger.info("Tokenizing and batching datasets...")
        loaders = {split: self._tokenize_and_collate(dataset[split]) for split in splits}
        if "train" in loaders:
            loaders["train"].shuffle = True

        num_train_epochs = self.num_train_epochs
        num_train_steps = num_train_epochs * len(loaders["train"])
        optimizer = AdamW(self.model.parameters(), lr=float(self.lr))
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )

        self.model, optimizer = self.accelerator.prepare(self.model, optimizer)
        for k, loader in loaders.items():
            loaders[k] = self.accelerator.prepare(loader)

        logger.info(f"Training for {num_train_epochs} epochs with {num_train_steps} steps.")
        progress_bar = tqdm(range(num_train_steps))
        for epoch in range(num_train_epochs):
            self.model.train()
            for i, batch in enumerate(loaders["train"]):
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            if "validation" in loaders:
                self.model.eval()
                running_loss = 0.0
                for dev_batch in loaders["validation"]:
                    with torch.no_grad():
                        outputs = self.model(**dev_batch)
                    true_preds, true_labels = self._process_outputs_for_eval(outputs)

                val_loss = running_loss / len(loaders["validation"])
                progress_bar.set_description(f"Validation loss: {val_loss}")
                self.model.train()

class WikiNER(WikiModelFromConfig):
    def __init__(self, config, tokenizer, label_set, **kwargs):
        super().__init__(config, tokenizer, **kwargs)
        self.label_set = label_set
        self.label_to_id = {label: i for i, label in enumerate(self.label_set)}
        self.id_to_label = {i: label for i, label in enumerate(self.label_set)}
        self.collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config["from_pretrained"],
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )
        self.model = self.model.to(self.device, dtype=self.torch_dtype)

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

    def train(self, dataset):

        splits = dataset.keys()
        logger.info("Tokenizing and batching datasets...")
        loaders = {split: self._tokenize_and_collate(dataset[split]) for split in splits}
        if "train" in loaders:
            loaders["train"].shuffle = True

        num_train_epochs = self.num_train_epochs
        num_train_steps = num_train_epochs * len(loaders["train"])
        optimizer = AdamW(self.model.parameters(), lr=float(self.lr))
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )

        self.model, optimizer = self.accelerator.prepare(self.model, optimizer)
        for k, loader in loaders.items():
            loaders[k] = self.accelerator.prepare(loader)

        logger.info(f"Training for {num_train_epochs} epochs with {num_train_steps} steps.")
        progress_bar = tqdm(range(num_train_steps))
        for epoch in range(num_train_epochs):
            self.model.train()
            for i, batch in enumerate(loaders["train"]):
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            if "validation" in loaders:
                self.model.eval()
                running_loss = 0.0
                for dev_batch in loaders["validation"]:
                    with torch.no_grad():
                        outputs = self.model(**dev_batch)
                    true_preds, true_labels = self._process_outputs_for_eval(outputs)

                val_loss = running_loss / len(loaders["validation"])
                progress_bar.set_description(f"Validation loss: {val_loss}")
                self.model.train()