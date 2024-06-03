import logging
from typing import List, Union

from transformers import PreTrainedTokenizerFast
from tokenizers import AddedToken, Tokenizer, normalizers, processors, pre_tokenizers
from ..utils.maps import TOKENIZER_PARAM_MAP as PARAM_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreTrainedTokenizerFast(PreTrainedTokenizerFast):

    """
    A tokenizer class that extends `tokenizer.PreTrainedTokenizerFast`.
    Adds a class method for training a tokenizer given a configuration and dataset.

    It is necessary to do this because `PreTrainedTokenizerFast`
    (or its parents class `PreTrainedTokenizer`) cannot be easily extended for training
    flexible tokenizers with a variety of (changing) parameters.

    Although it is possible to extend `tokenizers.implementations.base_tokenizer.Tokenizer`,
    that class is a Rust port and offers limited functionality within the Hugging Face ecosystem.
    This class provides a workaround for this limitation.

    While something like `FastTokenizerFromConfig` would be more appropriate, that is not
    a signature that is currently supported by the Hugging Face Tokenizer API and will throw
    a warning if loaded. In older versions of the library, it will throw an error. As such,
    overwriting the `PreTrainedTokenizerFast` signature is a temporary compromise.

    Methods
    -------
    train_from_config(
        dataset,
        model,
        normalizer,
        pre_tokenizer,
        decoder,
        post_processor,
        special_tokens,
        vocab_size,
        batch_size,
        **kwargs
    ):
        Trains a tokenizer based on provided configuration and dataset.
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
        model: str = "unigram",
        normalizer: Union[str, List[str]] = "nfkc",
        pre_tokenizer: Union[str, List[str]] = "metaspace",
        decoder: str = "metaspace",
        post_processor: bool = False,
        special_tokens: List[Union[str, AddedToken]] = None,
        vocab_size: Union[int, str] = "auto",
        batch_size: int = 1000,
        **kwargs
    ):

        """
        Train a tokenizer from a configuration.

        Parameters
        ----------
        dataset : list of str
            A list of strings representing the dataset.
        model : str, optional
            The model type for the tokenizer (default is "unigram").
        normalizer : Union[str, list of str], optional
            The normalizer(s) to use (default is "nfkc").
        pre_tokenizer : Union[str, list of str], optional
            The pre-tokenizer(s) to use (default is "metaspace").
        decoder : str, optional
            The decoder to use (default is "metaspace").
        post_processor : bool, optional
            Whether to use an MLM-based post-processor (default is False).
        special_tokens : list of Union[str, AddedToken], optional
            A list of special tokens to add (default is None).
        vocab_size : Union[int, str], optional
            The vocabulary size or "auto" to predict it (default is "auto").
        batch_size : int, optional
            The batch size for training (default is 1000).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        PreTrainedTokenizerFast
            An instance of the trained tokenizer.
        """

        logger.info("Building tokenizer from config:")
        tokenizer = Tokenizer(PARAM_MAP["model"][model]())
        tokenizer.add_special_tokens(list(special_tokens.values()))

        # Handle normalizer
        if isinstance(normalizer, str):
            tokenizer.normalizer = PARAM_MAP["normalizer"][normalizer]()
        else:
            tokenizer.normalizer = normalizers.Sequence(
                [PARAM_MAP["normalizer"][norm]() for norm in normalizer]
            )

        # Handle pre_tokenizer
        if isinstance(pre_tokenizer, str):
            tokenizer.pre_tokenizer = PARAM_MAP["pre_tokenizer"][pre_tokenizer]()
        else:
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [PARAM_MAP["pre_tokenizer"][pt]() for pt in pre_tokenizer]
            )

        tokenizer.decoder = PARAM_MAP["decoder"][decoder]()

        # Determine vocab_size
        if vocab_size == "auto":
            vocab_size = cls.predict_vocab_size(len("".join(dataset)))

        trainer = PARAM_MAP["trainer"][model](
            vocab_size=vocab_size,
            special_tokens=list(special_tokens.values()),
            unk_token=special_tokens["unk_token"]
        )

        if post_processor:
            tokenizer.post_processor = processors.TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
            )

        logging.info(f"Training tokenizer_cfg on {len(dataset)} samples...")
        tokenizer.train_from_iterator(
            cls.batch_iterator(dataset, batch_size=batch_size),
            trainer=trainer,
            length=len(dataset)
        )
        logging.info(f"Trained a tokenizer_cfg with vocab size: {tokenizer.get_vocab_size()}")

        return cls(tokenizer_object=tokenizer, **special_tokens, **kwargs)

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
