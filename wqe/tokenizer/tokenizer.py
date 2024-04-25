import json
import logging

from tokenizers import Tokenizer, processors, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from wqe.utils.maps import TOKENIZER_PARAM_MAP as PARAM_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastTokenizerFromConfig(PreTrainedTokenizerFast):

    @classmethod
    def train_from_config(cls, dataset, config, batch_size=1000):

        logger.info("Building tokenizer from config.")

        tokenizer = Tokenizer(
            PARAM_MAP["model"][config.model]()
        )

        tokenizer.add_special_tokens(list(config.special_tokens.values()))

        if isinstance(config.pre_tokenizer, list):
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [PARAM_MAP["pre_tokenizer"][pt]() for pt in config.pre_tokenizer]
            )
        else:
            tokenizer.pre_tokenizer = PARAM_MAP["pre_tokenizer"][config.pre_tokenizer]()
        tokenizer.decoder = PARAM_MAP["decoder"][config.decoder]()
        tokenizer.normalizer = PARAM_MAP["normalizer"][config.normalizer]()

        if config.vocab_size == "auto":
            config.vocab_size = cls.predict_vocab_size(len("".join(dataset["text"])))

        trainer = PARAM_MAP["trainer"][config.trainer](
            vocab_size=config.vocab_size,
            # min_frequency=config.min_frequency,
            special_tokens=list(config.special_tokens.values()),
            unk_token=config.unk_token
        )

        if config.post_processor:
            tokenizer.post_processor = processors.TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
            )

        logging.info(f"Training tokenizer on {len(dataset)} samples...")

        tokenizer.train_from_iterator(
            cls.batch_iterator(dataset, batch_size=batch_size),
            trainer=trainer,
            length=len(dataset)
        )

        logging.info(f"Trained a tokenizer with vocab size: {tokenizer.get_vocab_size()}")

        return cls(tokenizer_object=tokenizer, unk_id=0, **config.special_tokens)

    @staticmethod
    def predict_vocab_size(length_in_chars):

        k = 14.786406013978949
        b = 0.5817526745145639
        threshold_ratio = 0.027731383953247812

        return int(k * (length_in_chars ** b) * threshold_ratio)

    @staticmethod
    def batch_iterator(dataset, batch_size=1000):
        """
        Iterate over the dataset in batches.

        Args:
            dataset (datasets.Dataset): The dataset to iterate over.
            batch_size (int): The batch size.

        Returns:
            list: A list of batches of the dataset.
        """
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]["text"]
