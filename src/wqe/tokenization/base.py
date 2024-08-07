import logging
from typing import List

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, normalizers, pre_tokenizers

from ..utils.config import TokenizerConfig
from ..utils.maps import TOKENIZER_PARAM_MAP as PARAM_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTokenizerMixin:

    @staticmethod
    def predict_vocab_size(length_in_chars):

        """
        Predict the vocabulary size based on the length of the dataset in characters.
        Coefficients taken by fitting Heap's Law to a variety of Wiki datasets with
        gold tokenization. The threshold ratio is a heuristic value that was fitted
        to the mean vocabulary item frequency of the same data.

        Parameters
        ----------
        length_in_chars : int
            The length of the dataset in characters.

        Returns
        -------
        int
            The predicted vocabulary size.
        """

        k = 14.786406013978949
        b = 0.5817526745145639
        threshold_ratio = 0.027731383953247812
        return int(k * (length_in_chars ** b) * threshold_ratio)

    @staticmethod
    def batch_iterator(dataset, batch_size=10000):

        """
        Iterate over the dataset in batches.

        Parameters
        ----------
        dataset : list of str
            The dataset to iterate over.
        batch_size : int, optional
            The batch size (default is 1000).

        Yields
        ------
        Iterator[list of str]
            An iterator over batches of the dataset.
        """

        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]


class HfTokenizerFromConfig(PreTrainedTokenizerFast, BaseTokenizerMixin):

    """
    A tokenization class that extends `tokenization.HfTokenizerFromConfig`.
    Adds a class method for training a tokenization given a dataset and configuration.

    It is necessary to do this because `HfTokenizerFromConfig`
    (or its parents class `PreTrainedTokenizer`) cannot be easily extended for training
    flexible tokenizers with a variety of (changing) parameters.

    Although it is possible to extend `tokenizers.implementations.base_tokenizer.Tokenizer`,
    that class is a Rust port and offers limited functionality within the Hugging Face ecosystem.
    This class provides a workaround for this limitation.

    Methods
    -------
    train_from_config(
        dataset,
        config,
        **kwargs
    ):
        Trains a tokenization based on provided dataset and configuration.
        Components generally follow the huggingface `Tokenizers` API,
        though full cross-compatibility is not guaranteed.

    predict_vocab_size(length_in_chars):
        Predicts the vocabulary size based on the length of the dataset in characters.

    batch_iterator(dataset, batch_size):
        Yields batches of the dataset for training.
    """

    @classmethod
    def train_from_config(
            cls,
            dataset: List[str],
            config: TokenizerConfig,
            **kwargs
    ):

        """
        Train a tokenization from a configuration.

        Parameters
        ----------
        dataset : list of str
            A list of documents representing the dataset.
        config : TokenizerConfig
            A configuration object for the tokenization.
            See `wqe.utils.config.TokenizerConfig` for details.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        PreTrainedTokenizerFast
            An instance of the trained tokenization.
        """

        logger.info("Building tokenization from config:")

        tokenizer = Tokenizer(PARAM_MAP["model"][config.model.type](**config.model.args))
        special_tokens = config.special_tokens
        tokenizer.add_special_tokens(list(special_tokens.values()))

        if isinstance(config.normalizer, list):
            tokenizer.normalizer = normalizers.Sequence(
                [PARAM_MAP["normalizer"][norm.type](**norm.args) for norm in config.normalizer]
            )
        else:
            tokenizer.normalizer = \
                PARAM_MAP["normalizer"][config.normalizer.type](**config.normalizer.args)

        if isinstance(config.pre_tokenizer, list):
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [PARAM_MAP["pre_tokenizer"][pt.type](**pt.args) for pt in config.pre_tokenizer]
            )
        else:
            tokenizer.pre_tokenizer = \
                PARAM_MAP["pre_tokenizer"][config.pre_tokenizer.type](**config.pre_tokenizer.args)

        tokenizer.decoder = PARAM_MAP["decoder"][config.decoder.type](**config.decoder.args)

        vocab_size = cls.predict_vocab_size(len("".join(dataset))) \
            if config.vocab_size == "auto" else config.vocab_size

        trainer = PARAM_MAP["trainer"][config.trainer.type](
            vocab_size=vocab_size,
            special_tokens=list(special_tokens.values()),
            unk_token=special_tokens["unk_token"],
            **config.trainer.args
        )

        logging.info(f"Training {config.model.type} tokenizer on {len(dataset)} samples...")

        tokenizer.train_from_iterator(
            cls.batch_iterator(dataset, batch_size=config.batch_size),
            trainer=trainer,
            length=len(dataset)
        )

        logging.info(f"Trained a tokenizer_cfg with vocab size: {tokenizer.get_vocab_size()}")

        return cls(tokenizer_object=tokenizer, **special_tokens, **kwargs)
