import logging
import math

from typing import Dict, Optional, Union

import torch

from datasets import Dataset, DatasetDict
from datasets.utils.logging import set_verbosity_error
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader
from transformers import (
    CONFIG_MAPPING,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast
)

from .base import ModelFromConfig
from ..tokenizer.tokenizer import FastTokenizerFromConfig
from ..utils.config import TrainingParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_verbosity_error()


class MLM(ModelFromConfig):

    """
    Class for Masked Language Model (MLM) training and evaluation.
    Works with BERT, RoBERTa, and DeBERTa models out of the box,
    but can be extended to other models.

    Parameters
    ----------
    load_path : str
        Path to load the model from (used in `_init_model_and_tokenizer`).
        If the path ends with ".json", the model will be initialized from a local config file.
        TODO: Make this check more robust.
    config : TrainingParameters
        Configuration object for training parameters.
        See `wqe.utils.config.TrainingParameters` for more details.
    checkpoint_path : str, optional
        Path to save model checkpoints during training (default is None).

    Methods
    -------
    _init_model_and_tokenizer(dataset=None, tokenizer=None)
        Initializes the model and tokenizer.
    _tokenize_and_collate(dataset)
        Tokenizes and collates a dataset into a PyTorch DataLoader.
    _eval_loop(loader)
        Performs an evaluation loop on the given DataLoader and returns loss and perplexity scores.
    """

    def __init__(
            self,
            load_path: str,
            config: TrainingParameters,
            **kwargs
    ):

        super().__init__(load_path, config, **kwargs)

    def _init_model_and_tokenizer(
            self,
            dataset: DatasetDict = None,
            tokenizer: Optional[Union[PreTrainedTokenizerFast, FastTokenizerFromConfig]] = None
    ):

        """
        Initializes the model and tokenizer for MLM.
        If model was initialized with a local config file, the tokenizer must be provided.

        Parameters
        ----------
        dataset : DatasetDict, optional
            The dataset to use for initializing the model and tokenizer.
            Generally not needed here, as no labels are required.
        tokenizer : PreTrainedTokenizerFast or FastTokenizerFromConfig, optional
            The tokenizer to use for the model.
            If not provided, the tokenizer will be loaded from the hub.
        """

        if self.load_path.endswith(".json"):
            assert tokenizer is not None, "Tokenizer must be provided when training from scratch."

            self.tokenizer = tokenizer
            self.tokenizer.processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.tokenizer.cls_token_id),
                    ("[SEP]", self.tokenizer.sep_token_id)
                ]
            )

            model_config = CONFIG_MAPPING[self.model_type].from_json_file(self.load_path)
            model_config.vocab_size = self.tokenizer.vocab_size
            for special_token in self.tokenizer.special_tokens_map.keys():
                special_token_id = getattr(self.tokenizer, f"{special_token}_id")
                setattr(
                    model_config,
                    f"{special_token}_token_id",
                    special_token_id)

            logger.info(f"Initializing model with config: \n{model_config}")

            self._model = AutoModelForMaskedLM.from_config(model_config)

        else:
            if not tokenizer:
                logger.warning("Tokenizer not provided. Loading tokenizer from hub.")
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{self.load_path}")

            logger.info(f"Loading model from hub: {self.load_path}.")
            self._model = AutoModelForMaskedLM.from_pretrained(f"{self.load_path}")

        self._model = self._model.to(self.device, dtype=self.torch_dtype)

        logger.info(f"{self._model.config.model_type} for {self.task} loaded.")
        logger.info(f"Number of parameters: {round(self._model.num_parameters() / 1e6)}M")

        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mask_prob
        )

    def _tokenize_and_collate(
            self,
            dataset: Dataset
    ) -> DataLoader:

        """
        Tokenizes and collates a dataset into a PyTorch DataLoader.

        Parameters
        ----------
        dataset : Dataset
            The dataset to tokenize and collate.

        Returns
        -------
        DataLoader
            The PyTorch DataLoader for the tokenized and collated dataset.
        """

        batched_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                padding=self.padding_strategy,
                max_length=self.max_length,
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

    def _eval_loop(
            self,
            loader: DataLoader
    ) -> Dict[str, float]:

        """
        Performs an evaluation loop on the given DataLoader and returns loss and perplexity scores.
        Warning: perplexity should be taken with a grain of salt, as it is not well-defined for MLMs.

        Parameters
        ----------
        loader : DataLoader
            The PyTorch DataLoader to use for evaluation.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the 'loss' and 'perplexity' scores.
        """

        running_loss = []
        for batch in loader:
            with torch.no_grad():
                outputs = self._model(**batch)
            loss = outputs.loss
            running_loss.append(loss.item())
        eval_loss = math.fsum(running_loss) / len(running_loss)
        perplexity = math.exp(eval_loss)

        return {"loss": eval_loss, "perplexity": perplexity}


class CLM(MLM):

    """
    Class for Causal Language Model (CLM) training and evaluation.
    Inherits from MLM, as the only difference is the task type.

    Parameters
    ----------
    load_path : str
        Path to load the model from (used in `_init_model_and_tokenizer`).
        If the path ends with ".json", the model will be initialized from a local config file.
    config : TrainingParameters
        Configuration object for training parameters.
        See `wqe.utils.config.TrainingParameters` for more details.

    Methods
    -------
    _init_model_and_tokenizer(dataset=None, tokenizer=None)
        Initializes the model and tokenizer for Causal Language Modeling.
    """

    def __init__(
            self,
            load_path: str,
            config: TrainingParameters,
            **kwargs
    ):

        super().__init__(load_path, config, **kwargs)

    def _init_model_and_tokenizer(
            self,
            dataset: DatasetDict = None,
            tokenizer: Optional[Union[PreTrainedTokenizerFast, FastTokenizerFromConfig]] = None
    ):

        """
        Initializes the model and tokenizer for CLM.

        Parameters
        ----------
        dataset : DatasetDict, optional
            The dataset to use for initializing the model and tokenizer.
        tokenizer : PreTrainedTokenizerFast or FastTokenizerFromConfig, optional
            The tokenizer to use for the model.
        """

        if self.load_path.endswith(".json"):
            assert tokenizer is not None, "Tokenizer must be provided when training from scratch."

            self.tokenizer = tokenizer
            model_config = CONFIG_MAPPING[self.model_type].from_json_file(self.load_path)
            model_config.vocab_size = self.tokenizer.vocab_size

            for special_token in self.tokenizer.special_tokens_map.keys():
                special_token_id = getattr(self.tokenizer, f"{special_token}_id")
                setattr(
                    model_config,
                    f"{special_token}_token_id",
                    special_token_id)

            logger.info(f"Initializing model with config: \n{model_config}")

            self._model = AutoModelForCausalLM.from_config(model_config)
        else:
            if not tokenizer:
                logger.warning("Tokenizer not provided. Loading tokenizer from hub.")
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{self.load_path}")

            logger.info(f"Loading model from hub: {self.load_path}.")
            self._model = AutoModelForCausalLM.from_pretrained(f"{self.load_path}")

        self._model = self._model.to(self.device, dtype=self.torch_dtype)

        logger.info(f"{self._model.config.model_type} for {self.task} loaded.")
        logger.info(f"Number of parameters: {round(self._model.num_parameters() / 1e6)}M")

        self.collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
