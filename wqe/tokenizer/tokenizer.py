import logging
from typing import Iterator, List, Union
import json
import pandas as pd
import sentencepiece as spm
from sentencepiece import SentencePieceTrainer
import tokenizers
from tokenizers import (
    AddedToken,
    Tokenizer,
    Regex,
    normalizers,
    processors,
    pre_tokenizers,
    decoders,
    trainers
)
from tokenizers.implementations.base_tokenizer import BaseTokenizer

from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from transformers import DebertaV2Tokenizer

from utils.maps import TOKENIZER_PARAM_MAP as PARAM_MAP

# from ..utils.maps import TOKENIZER_PARAM_MAP as PARAM_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreTrainedTokenizerFast(PreTrainedTokenizerFast):

    @classmethod
    def train_from_config(cls, dataset, config, batch_size=1000, **kwargs):

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
        if isinstance(config.pre_tokenizer, list):
            tokenizer.normalizer = normalizers.Sequence(
                [PARAM_MAP["normalizer"][pt]() for pt in config.normalizer]
            )
        else:
            tokenizer.normalizer = PARAM_MAP["normalizer"][config.normalizer]()

        if config.vocab_size == "auto":
            config.vocab_size = cls.predict_vocab_size(len("".join(dataset["text"])))

        trainer = PARAM_MAP["trainer"][config.trainer](
            vocab_size=config.vocab_size,
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

        return cls(tokenizer_object=tokenizer, unk_id=1, **config.special_tokens, **kwargs)

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

class SentencePieceTokenizer:
    def __init__(self, config):
        pass

    def train_from_iterator(self, dataset, config):
        vocab_size = self.predict_vocab_size(len("".join(dataset["text"])))
        iterator = self.batch_iterator(dataset)

        spm.SentencePieceTrainer.Train(
            sentence_iterator=iterator,
            model_prefix="sentencepiece",
            vocab_size=vocab_size,
            model_type="unigram",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='[PAD]',
            unk_piece='[UNK]',
            bos_piece='[CLS]',
            eos_piece='[SEP]',
            user_defined_symbols=["[MASK]"],
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            split_by_unicode_script=False,
            split_by_number=False,
            split_by_whitespace=True,
            split_digits=False,
            allow_whitespace_only_pieces=False)

        with open("sentencepiece.vocab", "r", encoding="utf-8") as handle:
            vocab = []
            for line in handle:
                parts = line.rstrip().split("\t")
                if len(parts) != 2 or not parts[1].isnumeric():
                    print("Strange line detected:", line.__repr__())
                    continue

                vocab.append((parts[0], parts[1]))


        return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
            tokenizers.model.Unigram(vocab, unk_id=1, byte_fallback=False)))

    def predict_vocab_size(self, length_in_chars):

            k = 14.786406013978949
            b = 0.5817526745145639
            threshold_ratio = 0.027731383953247812

            return int(k * (length_in_chars ** b) * threshold_ratio)
    def batch_iterator(self, dataset, batch_size=1000):
        """
        Iterate over the dataset in batches.

        Args:
            dataset (datasets.Dataset): The dataset to iterate over.
            batch_size (int): The batch size.

        Returns:
            list: A list of batches of the dataset.
        """
        for i in range(0, len(dataset)):
            yield dataset[i]["text"]


