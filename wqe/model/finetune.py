import logging

# import numpy as np
import torch
import wandb

from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm import tqdm
from transformers import get_scheduler, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from torch.optim import AdamW

from collections import Counter

from .model import ModelFromConfig
from utils.maps import TASK_TO_MODEL_AND_COLLATOR_MAPPING
# from .utils.maps import TASK_TO_MODEL_AND_COLLATOR_MAPPING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenericModelForFineTuning(ModelFromConfig):

    def __init__(self,
                 config,
                 label_set,
                 load_path,
                 task,
                 **kwargs
                 ):

        super().__init__(config, **kwargs)

        self.label_set = label_set
        self.label_to_id = {label: i for i, label in enumerate(self.label_set)}
        self.id_to_label = {i: label for i, label in enumerate(self.label_set)}
        self.task = task

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(load_path)
        self.collator = TASK_TO_MODEL_AND_COLLATOR_MAPPING[self.task]["collator"](tokenizer=self.tokenizer)
        self.model = TASK_TO_MODEL_AND_COLLATOR_MAPPING[self.task]["model"].from_pretrained(
            load_path,
            num_labels=len(self.label_set),
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )
        self.model = self.model.to(self.device, dtype=self.torch_dtype)

    def _tokenize_and_collate(self, dataset):

        raise NotImplementedError("Subclasses must implement this method.")

    def _eval_loop(self, loader, split="validation"):

        raise NotImplementedError("Subclasses must implement this method.")

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
                if wandb:
                    wandb.log({"loss": loss.item()})
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            if "validation" in loaders:
                scores = self._eval_loop(loaders["validation"], split="validation")
                wandb.log(scores)
                if epoch+1 == num_train_epochs:
                    wandb.run.summary.update(scores)

        if "test" in loaders:
            scores = self._eval_loop(loaders["test"], split="test")
            wandb.run.summary.update(scores)


class Tagger(GenericModelForFineTuning):
    def __init__(self, config, **kwargs):
        super().__init__(config, task="tagger", **kwargs)

    def _align_labels(self, example):
        tokenized_input = self.tokenizer(
            example["tokens"],
            is_split_into_words=True,
        )

        seen = []
        labels = []
        for idx in tokenized_input.word_ids()[1:-1]:
            seen.append(idx)
            if Counter(seen)[idx] == 2:
                labels.append(example["tags"][idx])
            else:
                labels.append(-100)

        # seen = set()
        # labels = []
        # for idx in tokenized_input.word_ids()[1:-1]:
        #     if idx in seen:
        #         labels.append(-100)
        #     else:
        #         labels.append(example["tags"][idx])
        #         seen.add(idx)


                

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
            batch_size=self.batch_size
        )

        return loader

    def _eval_loop(self, loader, split="validation"):

        self.model.eval()

        results = {"preds": [], "labels": []}
        for batch in loader:
            with torch.no_grad():
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                predictions = self.accelerator.pad_across_processes(
                    predictions, dim=1, pad_index=-100
                )
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100
                )

                predictions = self.accelerator.gather(predictions).detach().cpu().clone().numpy()
                labels = self.accelerator.gather(labels).detach().cpu().clone().numpy()

                true_labels = [lab for label in labels for lab in label if lab != -100]
                true_preds = [
                    pred for prediction, label in zip(predictions, labels)
                    for pred, lab in zip(prediction, label) if lab != -100
                ]

                results["preds"].extend(true_preds)
                results["labels"].extend(true_labels)

        # report = classification_report(
        #     results["labels"],
        #     results["preds"],
        #     labels=[i for i in range(len(self.label_set))],
        #     target_names=self.label_set,
        #     zero_division=0.0
        # )

        scores = precision_recall_fscore_support(
            results["labels"],
            results["preds"],
            labels=[i for i in range(len(self.label_set))],
            zero_division=0.0,
            average="weighted"
        )

        score_dict = {f"{split}_{k}": v for (k, v) in zip(["precision", "recall", "f1"], scores[:3])}

        return score_dict


class Classifier(GenericModelForFineTuning):
    def __init__(self, config, **kwargs):
        super().__init__(config, task="classifier", **kwargs)

    def _tokenize_and_collate(self, dataset):
        batched_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            ),
            batched=True,
            remove_columns=["text"]
        )

        loader = DataLoader(
            batched_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=True
        )

        return loader

    def _eval_loop(self, loader, split="validation"):

        self.model.eval()

        results = {"preds": [], "labels": []}
        for batch in loader:
            with torch.no_grad():
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                predictions = self.accelerator.pad_across_processes(
                    predictions, dim=1, pad_index=-100
                )
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100
                )

                results["preds"].extend(self.accelerator.gather(predictions).detach().cpu().clone().numpy().tolist())
                results["labels"].extend(self.accelerator.gather(labels).detach().cpu().clone().numpy().tolist())

        # report = classification_report(
        #     results["labels"],
        #     results["preds"],
        #     labels=[i for i in range(len(self.label_set))],
        #     target_names=self.label_set,
        #     zero_division=0.0
        # )

        scores = precision_recall_fscore_support(
            results["labels"],
            results["preds"],
            labels=[i for i in range(len(self.label_set))],
            zero_division=0.0,
            average="weighted"
        )

        score_dict = {f"{split}_{k}": v for (k, v) in zip(["precision", "recall", "f1"], scores[:3])}

        return score_dict