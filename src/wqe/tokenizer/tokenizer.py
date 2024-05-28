import logging
import os
from typing import Iterator, List, Union

from transformers import PreTrainedTokenizerFast
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers import (
    AddedToken,
    Tokenizer,
    normalizers,
    processors,
    pre_tokenizers,
    trainers
)

from ..utils.maps import TOKENIZER_PARAM_MAP as PARAM_MAP


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenizerFromConfig(BaseTokenizer):

    def __init__(
        self,
        model: str = "unigram",
        normalizer: Union[str, List[str]] = "nkfc",
        pre_tokenizer: Union[str, List[str]] = "metaspace",
        decoder: str = "metaspace",
        post_processor: bool = False,
        **kwargs
    ):

        tokenizer = Tokenizer(
            PARAM_MAP["model"][model]()
        )

        if isinstance(normalizer, list):
            tokenizer.normalizer = normalizers.Sequence(
                [PARAM_MAP["normalizer"][norm]() for norm in normalizer]
            )
        else:
            tokenizer.normalizer = PARAM_MAP["normalizer"][normalizer]()

        if isinstance(pre_tokenizer, list):
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [PARAM_MAP["pre_tokenizer"][pt]() for pt in pre_tokenizer]
            )
        else:
            tokenizer.pre_tokenizer = PARAM_MAP["pre_tokenizer"][pre_tokenizer]()

        if post_processor:
            tokenizer.post_processor = processors.TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
            )

        tokenizer.decoder = PARAM_MAP["decoder"][decoder]()

        super().__init__(tokenizer, **kwargs)

    def train(
        self,
        examples: List[str],
        vocab_size: Union[int, str],
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = None,
        **kwargs
    ):
        if vocab_size == "auto":
            vocab_size = self.predict_vocab_size(len("".join(examples)))

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=show_progress
        )

        self._tokenizer.train(examples, trainer=trainer, **kwargs)

    def train_new_from_iterator(
        self,
        iterator: Iterator[str],
        vocab_size: Union[int, str],
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = None,
        **kwargs,
    ):
        if vocab_size == "auto":
            vocab_size = self.predict_vocab_size(len("".join(iterator)))

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=show_progress
        )

        self._tokenizer.train_from_iterator(iterator, trainer=trainer, **kwargs)

    @staticmethod
    def predict_vocab_size(length_in_chars):

        k = 14.786406013978949
        b = 0.5817526745145639
        threshold_ratio = 0.027731383953247812

        return int(k * (length_in_chars ** b) * threshold_ratio)

    def convert_to_fast_tokenizer(self):
        return PreTrainedTokenizerFast(tokenizer_object=self._tokenizer)

