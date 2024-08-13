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

