import logging

from typing import Union

import evaluate
import torch

from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast
)

from .base import ModelFromConfig
from ..tokenization import HfTokenizerFromConfig
from ..utils.config import TrainingParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tagger(ModelFromConfig):

    """
    Class for token-level classification tasks, such as
    Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.
    Only NER and POS are supported for now.

    Parameters
    ----------
    load_path : str
        Path to load the model from (either a local path or a Hugging Face Hub path).
    config : TrainingParameters
        Configuration object for training parameters.
    **kwargs
        Additional keyword arguments for the parent class.

    Attributes
    ----------
    label_set : List[str]
        List of labels for the task.
        Assumes the `tags` feature in the dataset.
    metrics : evaluate.EvaluationModule
        Evaluation metrics for the task.

    Methods
    -------
    _init_model_and_tokenizer(dataset=None, tokenization=None)
        Initializes the model and tokenization for the task.
    _init_metrics()
        Initializes the evaluation metrics for the task.
    _align_labels(example)
        Aligns the token-level labels with the tokenized input.
    _tokenize_and_collate(dataset)
        Tokenizes and collates a dataset into a PyTorch DataLoader.
    _eval_loop(loader)
        Performs an evaluation loop on the given DataLoader and returns the evaluation scores.
    _eval_loop_ner(loader)
        Performs an evaluation loop for the NER task.
    _eval_loop_pos(loader)
        Performs an evaluation loop for the POS tagging task.
    """

    def __init__(
            self,
            load_path: str,
            config: TrainingParameters,
            **kwargs
    ):

        super().__init__(load_path, config, **kwargs)

        self._init_metrics()

    def _init_model_and_tokenizer(
            self,
            dataset: DatasetDict = None,
            tokenizer: Union[PreTrainedTokenizerFast, HfTokenizerFromConfig] = None
    ):

        """
        Initialize the model and tokenization for tagging.

        Parameters
        ----------
        dataset : DatasetDict, optional
            The dataset used for the task.
            Assumes the `tags` feature in the dataset.
        tokenizer : Union[PreTrainedTokenizerFast, HfTokenizerFromConfig], optional
            The tokenization to be used. Generally not needed, as the tokenization will be loaded
            from the same path as the model.
        """

        self.label_set = dataset["train"].features["tags"].feature.names

        self._model = AutoModelForTokenClassification.from_pretrained(
            self.load_path,
            num_labels=len(self.label_set),
            id2label={i: label for i, label in enumerate(self.label_set)},
            label2id={label: i for i, label in enumerate(self.label_set)}
        )

        self.tokenizer = tokenizer if tokenizer else PreTrainedTokenizerFast.from_pretrained(self.load_path)
        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({"pad_token":"[PAD]"})
        self.collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        logger.info(f"{self._model.config.model_type} for {self.task} loaded.")
        logger.info(f"Number of parameters: {round(self._model.num_parameters() / 1e6)}M")

    def _init_metrics(self):

        """
        Initialize the evaluation metrics for the task.
        """

        if self.task == "ner":
            self.metrics = evaluate.load("seqeval")
        elif self.task == "pos":
            # Not adding f1 because, at time of writing, the zero_division argument
            # is not supported in `evaluate` for f1, despite being implemented
            # in the underlying `sklearn` function. It will be computed manually in
            # the `_eval_loop_pos` method.
            self.metrics = evaluate.combine(["precision", "recall"])
        else:
            raise ValueError(f"Task {self.task} not supported. Only 'ner' and 'pos' are supported for now.")

    def _align_labels(self, example):

        """
        Align the token-level labels with the tokenized input.
        Have to ignore special tokens and common prefixes/suffixes.

        Parameters
        ----------
        example : dict
            A single example from the dataset.

        Returns
        -------
        dict
            The tokenized input with aligned labels.
        """

        TO_IGNORE = ["Ġ", "▁", "##", "Ċ"] + list(self.tokenizer.special_tokens_map.values())

        tokenized_input = self.tokenizer(
            example["tokens"],
            padding=self.padding_strategy,
            max_length=self.max_length,
            truncation=True,
            is_split_into_words=True
        )

        seen = set()
        labels = []
        for i, word_id in enumerate(tokenized_input.word_ids()):
            token_id = tokenized_input["input_ids"][i]
            tag_id = example["tags"][word_id] if word_id is not None else -100
            if self.tokenizer.convert_ids_to_tokens(token_id) in TO_IGNORE:
                labels.append(-100)
            elif word_id in seen:
                labels.append(-100)
            else:
                labels.append(tag_id)
                seen.add(word_id)

        tokenized_input["labels"] = labels

        return tokenized_input

    def _tokenize_and_collate(self, dataset):

        """
        Tokenize and collate a dataset into a PyTorch DataLoader.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be tokenized and collated.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader containing the tokenized and collated dataset.
        """

        batched_dataset = dataset.map(
            self._align_labels,
            remove_columns=dataset.column_names
        )

        loader = DataLoader(
            batched_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            pin_memory=True
        )

        return loader

    def _eval_loop(self, loader):

        """
        Perform an evaluation loop on the given DataLoader and return scores.
        Have to differentiate between NER and POS tagging tasks since `seqeval` only supports NER.

        Parameters
        ----------
        loader : DataLoader
            A PyTorch DataLoader containing the data to be evaluated.

        Returns
        -------
        dict
            A dictionary containing the evaluation scores (precision, recall, f1).
        """

        if self.task == "ner":
            return self._eval_loop_ner(loader)
        elif self.task == "pos":
            return self._eval_loop_pos(loader)

    def _eval_loop_ner(self, loader):

        """
        Perform an evaluation loop for NER.

        Parameters
        ----------
        loader : DataLoader
            A PyTorch DataLoader containing the data to be evaluated.

        Returns
        -------
        dict
            A dictionary containing the evaluation scores (precision, recall, f1) for NER.
        """

        self._model.eval()

        for batch in loader:
            with torch.no_grad():
                outputs = self._model(**batch)

            preds = torch.argmax(outputs.logits, dim=-1)
            idx_to_keep = batch["labels"] != -100

            filtered_preds = [
                list(map(
                    self.label_set.__getitem__,
                    preds[i][idx_to_keep[i]].detach().cpu().tolist())
                )
                for i in range(len(preds))
            ]

            filtered_labels = [
                list(map(
                    self.label_set.__getitem__,
                    batch["labels"][i][idx_to_keep[i]].detach().cpu().tolist())
                )
                for i in range(len(preds))
            ]

            self.metrics.add_batch(predictions=filtered_preds, references=filtered_labels)

        scores = self.metrics.compute(zero_division=0.0)

        return {
            "precision": scores["overall_precision"],
            "recall": scores["overall_recall"],
            "f1": scores["overall_f1"]
        }

    def _eval_loop_pos(self, loader):

        """
        Perform an evaluation loop for POS.

        Parameters
        ----------
        loader : DataLoader
            A PyTorch DataLoader containing the data to be evaluated.

        Returns
        -------
        dict
            A dictionary containing the evaluation scores (precision, recall, f1) for POS.
        """

        self._model.eval()

        for batch in loader:

            with torch.no_grad():
                outputs = self._model(**batch)

            preds = torch.argmax(outputs.logits, dim=-1)
            idx_to_keep = batch["labels"] != -100

            filtered_preds = preds[idx_to_keep].detach().cpu().tolist()
            filtered_labels = batch["labels"][idx_to_keep].detach().cpu().tolist()

            self.metrics.add_batch(predictions=filtered_preds, references=filtered_labels)

        scores = self.metrics.compute(average="weighted", zero_division=0.0)

        return {
            "precision": scores["precision"],
            "recall": scores["recall"],
            # See note in `_init_metrics` method for why this is computed manually
            "f1": scores["precision"] * scores["recall"] / (scores["precision"] + scores["recall"]) * 2
        }


class Classifier(ModelFromConfig):

    """
    Class for sequence classification tasks.

    Parameters
    ----------
    load_path : str
        Path to load the model from (either a local path or a Hugging Face Hub path).
    config : TrainingParameters
        Configuration object for training parameters.
    **kwargs
        Additional keyword arguments for the parent class.

    Attributes
    ----------
    label_set : List[str]
        List of labels for the task.
        Assumes the `labels` feature in the dataset.
    metrics : evaluate.EvaluationModule
        Evaluation metrics for the task.

    Methods
    -------
    _init_model_and_tokenizer(dataset=None, tokenization=None)
        Initializes the model and tokenization for the task.
    _init_metrics()
        Initializes the evaluation metrics for the task.
    _tokenize_and_collate(dataset)
        Tokenizes and collates a dataset into a PyTorch DataLoader.
    _eval_loop(loader)
        Performs an evaluation loop on the given DataLoader and returns the evaluation scores.
    """

    def __init__(
            self,
            load_path: str,
            config: TrainingParameters,
            **kwargs
    ):

        super().__init__(load_path, config, **kwargs)

        self._init_metrics()

    def _init_model_and_tokenizer(
            self,
            dataset: DatasetDict = None,
            tokenizer: Union[PreTrainedTokenizerFast, HfTokenizerFromConfig] = None
    ):

        """
        Initialize the model and tokenization for classification.

        Parameters
        ----------
        dataset : DatasetDict, optional
            The dataset used for the task.
            Assumes the `labels` feature in the dataset.
        tokenizer : Union[PreTrainedTokenizerFast, HfTokenizerFromConfig], optional
            The tokenization to be used. Generally not needed, as the tokenization will be loaded
            from the same path as the model.
        """

        self.label_set = dataset["train"].features["labels"].names

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.load_path,
            num_labels=len(self.label_set),
            id2label={i: label for i, label in enumerate(self.label_set)},
            label2id={label: i for i, label in enumerate(self.label_set)}
        )

        self.tokenizer = tokenizer if tokenizer else PreTrainedTokenizerFast.from_pretrained(self.load_path)
        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({"pad_token":"[PAD]"})
        self.collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        logger.info(f"{self._model.config.model_type} for {self.task} loaded.")
        logger.info(f"Number of parameters: {round(self._model.num_parameters() / 1e6)}M")

    def _init_metrics(self):

        """
        Initialize the evaluation metrics for the task.
        """

        self.metrics = evaluate.combine(["precision", "recall"])

    def _tokenize_and_collate(self, dataset):

        """
        Tokenize and collate a dataset into a PyTorch DataLoader.
        Assumes `text` and `labels` are features in the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be tokenized and collated.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader containing the tokenized and collated dataset.
        """
        if "premise" in dataset.features:
            batched_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples["premise"], examples['hypothesis'],
                padding=self.padding_strategy,
                max_length=self.max_length,
                truncation=True
            ),
            batched=True,
            remove_columns=[column for column in dataset.column_names if column != "labels"]
        )
        else:
            batched_dataset = dataset.map(
                lambda examples: self.tokenizer(
                    examples["text"],
                    padding=self.padding_strategy,
                    max_length=self.max_length,
                    truncation=True
                ),
                batched=True,
                remove_columns=[column for column in dataset.column_names if column != "labels"]
            )

        batched_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        loader = DataLoader(
            batched_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            pin_memory=True
        )

        return loader

    def _eval_loop(self, loader):

        """
        Perform an evaluation loop on the DataLoader and return scores (precision, recall, f1).

        Parameters
        ----------
        loader : DataLoader
            A PyTorch DataLoader containing the data to be evaluated.

        Returns
        -------
        dict
            A dictionary containing the evaluation scores (precision, recall, f1).
        """

        self._model.eval()

        for batch in loader:
            with torch.no_grad():
                outputs = self._model(**batch)

            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            self.metrics.add_batch(predictions=preds, references=labels)

        scores = self.metrics.compute(average="weighted", zero_division=0.0)

        return {
            "precision": scores["precision"],
            "recall": scores["recall"],
            # See note in `_init_metrics` method for why this is computed manually
            "f1": scores["precision"] * scores["recall"] / (scores["precision"] + scores["recall"]) * 2
        }
    
