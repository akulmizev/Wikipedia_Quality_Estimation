import logging

from dataclasses import asdict
from typing import Dict, Optional, Union

import torch
import wandb

from datasets import Dataset, DatasetDict
from transformers import get_scheduler, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import Adafactor
from tqdm import tqdm

from .mixins import ModelInitMixin
from ..tokenizer.tokenizer import FastTokenizerFromConfig
from ..utils.config import TrainingParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFromConfig(ModelInitMixin):

    """
    Base class for loading and training a model from a configuration.

    Parameters
    ----------
    load_path : str
        The path to the model config file to load for training from scratch.
        Can also be the huggingface model string or path to a hub model,
        e.g. "bert-base-uncased" or "path/to/model".
    config : TrainingParameters
        Configuration for the training parameters.
        See `wqe.utils.config.TrainingParameters` for details.
    checkpoint_path : Union[str, None], optional
        Path to save the model checkpoint during training (default is None).

    Attributes
    ----------
    load_path : str
        Path to load the model from.
    _model : torch.nn.Module
        The model instance. Defined in subclasses.
    tokenizer : PreTrainedTokenizerFast or FastTokenizerFromConfig
        The tokenizer for the model. Defined in subclasses.
    collator : callable
        The collator function for the data loader. Defined in subclasses.
    """

    def __init__(
            self,
            load_path: str,
            config: TrainingParameters,
            checkpoint_path: Optional[Union[str, None]] = None
    ):
        super().__init__(**asdict(config), checkpoint_path=checkpoint_path)
        self.load_path = load_path
        self._model = None
        self.tokenizer = None
        self.collator = None
        
    @property
    def model(self):
        return self._model

    def __getattr__(self, item):
        return getattr(self._model, item)

    def _init_model_and_tokenizer(
            self,
            dataset: DatasetDict = None,
            tokenizer: Optional[Union[PreTrainedTokenizerFast, FastTokenizerFromConfig]] = None
    ):

        """
        Initializes the model and tokenizer. Also initializes the collator function, if applicable.
        This is heavily task-dependent and must be implemented in subclasses.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def _tokenize_and_collate(self, dataset: Dataset) -> DataLoader:

        """
        Tokenizes and collates the dataset into a data loader.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def _eval_loop(self, loader) -> Dict[str, float]:

        """
        Performs an evaluation loop on the given data loader.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def _prepare_for_training(
            self,
            dataset: DatasetDict
    ) -> Dict[str, DataLoader]:

        """
        Prepares the model, optimizers, and schedulers for training.

        Parameters
        ----------
        dataset : DatasetDict
            The dataset to use for training.

        Returns
        -------
        Dict[str, DataLoader]
            A dictionary containing the data loaders for different splits
            (e.g., 'train', 'validation', 'test').

        Raises
        ------
        ValueError
            If the 'train' split is not present in the dataset.
        """

        splits = dataset.keys()
        if "train" not in splits:
            raise ValueError("Train split must be present in the dataset.")

        logger.info("Tokenizing and batching datasets.")

        loaders = {split: self._tokenize_and_collate(dataset[split]) for split in splits}

        self.num_train_steps = self.num_train_epochs * len(loaders["train"])

        self.num_eval_steps = len(loaders["train"]) if self.num_eval_steps is None else self.num_eval_steps

        self.optimizer = AdamW(
            self._model.parameters(),
            lr=float(self.lr),
            weight_decay=0.05
        )

        # self.optimizer = Adafactor(
        #     self._model.parameters(),
        #     lr=self.lr,
        #     eps=(1e-30, 1e-3),
        #     clip_threshold=1.0,
        #     decay_rate=-0.8,
        #     beta1=None,
        #     weight_decay=0.0,
        #     relative_step=False,
        #     scale_parameter=False,
        #     warmup_init=False,
        # )

        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=self.num_train_steps // self.grad_accumulation_steps
        )

        self._model, self.optimizer = self.accelerator.prepare(self._model, self.optimizer)
        for k, loader in loaders.items():
            loaders[k] = self.accelerator.prepare(loader)

        self.accelerator.register_for_checkpointing(self.scheduler)
        self._model.gradient_checkpointing_enable()

        return loaders

    def _get_average_loss(
            self,
            loader: DataLoader
    ) -> float:

        """
        Calculates the average loss over the given data loader.
        Primarily used for checkpointing.

        Parameters
        ----------
        loader : DataLoader
            The data loader to use for calculating the average loss.

        Returns
        -------
        float
            The average loss over the data loader.
        """

        running_loss = 0.0
        for batch in loader:
            with torch.no_grad():
                outputs = self._model(**batch)
                loss = outputs.loss
                running_loss += loss.item()
        eval_loss = running_loss / len(loader)
        return eval_loss

    def train(
            self,
            dataset: DatasetDict,
            tokenizer: Optional[Union[PreTrainedTokenizerFast, FastTokenizerFromConfig]] = None,
            eval_split: str = "validation"
    ):

        """
        Trains the model on the provided dataset using a generic training loop.
        Checkpoints the model at the end of each epoch if a checkpoint path is provided.
        Uses `eval_split` for evaluation during training, as well as checkpointing.
        If `eval_split` is not present in the dataset, saves model at the end of each epoch.
        Otherwise, saves model at the epoch with the lowest loss on the evaluation split.

        Parameters
        ----------
        dataset : DatasetDict
            The dataset to use for training and evaluation.
            Can be a wqe.data.loader.WikiLoader instance.
        tokenizer : PreTrainedTokenizerFast or FastTokenizerFromConfig, optional
            The tokenizer to use for the model.
            Should only be provided if training from scratch with a config.
            If not provided, tries to load the tokenizer via the model string, e.g. "bert-base-uncased".
        eval_split : str, optional
            The split to use for evaluation during training (default is 'validation').
        """

        self._init_model_and_tokenizer(dataset=dataset, tokenizer=tokenizer)
        loaders = self._prepare_for_training(dataset)
        running_loss = torch.inf
        progress_bar = tqdm(range(self.num_train_steps))

        logger.info(f"Training model for {self.num_train_epochs} epoch(s) ({self.num_train_steps} steps).")
        logger.info(f"{self.batch_size} examples per batch, {self.grad_accumulation_steps} grad. accumulation steps.")

        for epoch in range(self.num_train_epochs):
            self._model.train()
            with self.accelerator.accumulate(self._model):
                for step, batch in enumerate(loaders["train"], start=1):
                    outputs = self._model(**batch)
                    loss = outputs.loss

                    loss_str = f"Step {step + (epoch * len(loaders['train']))} | Loss: {loss.item():.4f}"
                    progress_bar.set_description(loss_str)
                    progress_bar.update(1)
                    if self.wandb:
                        wandb.log({"train": {"loss": loss.item()}})

                    loss = loss / self.grad_accumulation_steps

                    self.accelerator.backward(loss)
                    if step % self.grad_accumulation_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                    if (step + (epoch * len(loaders['train']))) % self.num_eval_steps == 0:
                        if eval_split not in loaders:
                            logger.warning(f"No {eval_split} split found. Skipping evaluation.")
                            if self.checkpoint_path:
                                logger.info(f"Saving model checkpoint at epoch {epoch}.")
                                self.accelerator.save_state(self.checkpoint_path)
                        else:
                            scores = self._eval_loop(loaders[eval_split])
                            scores_str = " | ".join([f"val. {k}: {v:.4f}" for k, v in scores.items()])
                            logger.info(f"Step {step + (epoch * len(loaders['train']))} | {scores_str}")
                            
                            if self.checkpoint_path:
                                if scores["loss"] < running_loss:
                                    logger.info(f"Saving model checkpoint at epoch {epoch}.")
                                    self.accelerator.save_state(self.checkpoint_path)
                                    running_loss = scores["loss"]

                            if self.wandb:
                                wandb.log({"val": scores})
                            self._model.train()

        progress_bar.close()
        logger.info("Training complete.")
        if self.checkpoint_path:
            logger.info(f"Loading best model from {self.checkpoint_path}.")
            self._model = self.accelerator.load_state(self.checkpoint_path)

    def test(
            self,
            dataset: DatasetDict,
            split: str = "test",
            output_file: Optional[str] = None
    ):

        """
        Evaluates the model on the given dataset split.

        Parameters
        ----------
        dataset : DatasetDict
            The dataset to use for evaluation.
        split : str, optional
            The split to use for evaluation (default is 'test').
        output_file : str, optional
            The path to save the model predictions to (default is None).
        """

        logger.info(f"Running evaluation on {split} split...")
        loader = self._tokenize_and_collate(dataset[split])
        loader = self.accelerator.prepare(loader)
        scores = self._eval_loop(loader)
        logger.info(" | ".join([f"{k}: {v:.4f}" for k, v in scores.items()]))
        if self.wandb:
            wandb.log({"test": scores})

        if output_file:
            logger.info(f"Saving predictions to {output_file}.")
            with open(output_file, "w") as f:
                f.write("\n".join([f"{k}\t{v}" for k, v in scores.items()]))

    def save(self, path: str):

        """
        Saves the model and optimizer state to the specified path.

        Parameters
        ----------
        path : str
            The path to save the model to.
        """

        logger.info(f"Saving model to {path}.")
        self.accelerator.save_state(self.checkpoint_path)
