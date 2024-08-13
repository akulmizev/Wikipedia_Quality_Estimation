import unicodedata
import re

from typing import Iterator, List


def compute_ngrams(
    words: List[str],
    n: int = 3
) -> Iterator[str]:

    for gram in zip(*[words[i:] for i in range(n)]):
        yield ' '.join(gram)


def tokenize(
    text: str
) -> List[str]:

    return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)


def get_all_punctuation():
    punctuation = []
    for codepoint in range(0x110000):  # Unicode range
        char = chr(codepoint)
        category = unicodedata.category(char)
        if category.startswith('P'):
            punctuation.append(char)
    return ''.join(punctuation)
