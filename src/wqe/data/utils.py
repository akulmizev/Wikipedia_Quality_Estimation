import logging
import regex as re
import unicodedata

from functools import wraps
from typing import Dict, Iterator, List, Pattern, Union

import datasets
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def compute_ngrams(
    words: List[Union[str, bytes]],
    n: int = 3
) -> Iterator[str]:

    for gram in zip(*[words[i:] for i in range(n)]):
        yield ' '.join(gram)


def tokenize(
    text: str
) -> List[str]:

    # return re.findall(WHITE_SPACE, text, re.UNICODE)
    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)


def batch_tokenize(
        dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizerFast = None,
        batch_size: int = 1000
) -> datasets.Dataset:

    if tokenizer is None:
        tokenize_fn = tokenize
    else:
        tokenize_fn = tokenizer.tokenize

    tokenized_dataset = dataset.map(
        lambda x: {
            "tokens": tokenize_fn(x["text"])
        },
        batched=True,
        batch_size=batch_size
    )

    return tokenized_dataset


def c4_filter(
        line: str,
        patterns: Dict[str, Pattern]
) -> bool:
    """
    Filter for Common Crawl C4 dataset.
    """

    if not line.strip():
        return False

    if re.search(patterns["terminal_punct"], line[-1]) is None:
        return False

    if len(re.findall(patterns["tokens"], line)) < 5:
        return False

    if "lorem ipsum" in line.lower():
        return False

    return True


def get_all_punctuation():
    punctuation = []
    for codepoint in range(0x110000):  # Unicode range
        char = chr(codepoint)
        category = unicodedata.category(char)
        if category.startswith('P'):
            punctuation.append(char)
    return ''.join(punctuation)


def measure_deletion(func):
    @wraps(func)
    def wrapper(
        self,
        dataset,
        **kwargs
    ):

        initial_docs = len(dataset)
        full_text = "".join(dataset["text"])
        initial_chars = len(full_text)
        initial_bytes = len(full_text.encode("utf-8"))

        result_dataset = func(self, dataset, **kwargs)

        final_docs = len(result_dataset)
        full_text = "".join(result_dataset["text"])
        final_chars = len(full_text)
        final_bytes = len(full_text.encode("utf-8"))

        logger.info(
            f"Deleted: {initial_docs - final_docs} docs "
            f"({(initial_docs - final_docs) / initial_docs:.2%}), "
            f"{initial_chars - final_chars} chars "
            f"({(initial_chars - final_chars) / initial_chars:.2%}), "
            f"{(initial_bytes - final_bytes) / (1024 * 1024):.2f} MB "
            f"({(initial_bytes - final_bytes) / initial_bytes:.2%})"
        )

        return result_dataset

    return wrapper
