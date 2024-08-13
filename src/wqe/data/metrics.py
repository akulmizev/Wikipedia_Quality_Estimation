from typing import Any

from .utils import compute_ngrams, tokenize


class BaseMetric:
    NEEDS_TOKENIZER: bool = False
    HIGHER_IS_BETTER: bool = True
    ANNOTATION_TAG: str = "default"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> int:
        raise NotImplementedError("Metric not implemented. Please use a subclass.")


class LengthCharacters(BaseMetric):
    ANNOTATION_TAG = "length_chars"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> int:
        return len(example)


class LengthWords(BaseMetric):
    ANNOTATION_TAG = "length_words"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> int:
        return len(tokenize(example))


class LengthSubwords(BaseMetric):
    NEEDS_TOKENIZER = True
    ANNOTATION_TAG = "length_subwords"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> int:
        tokenize_fn = kwargs.get('tokenize_fn')
        if tokenize_fn is None:
            raise ValueError("This metric requires a tokenize function.")
        return len(tokenize_fn(example))


class UniqueWords(BaseMetric):
    ANNOTATION_TAG = "unique_words"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> int:
        words = [word[0] for word in tokenize(example)]
        return len(set(words))


class UniqueTrigrams(BaseMetric):
    ANNOTATION_TAG = "unique_trigrams"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> int:
        tokens = tokenize(example)
        trigrams = list(compute_ngrams(tokens, 3))
        return len(set(trigrams))


class UniqueCharacters(BaseMetric):
    ANNOTATION_TAG = "unique_chars"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> int:
        return len(set(example))


class UniqueCharacterTrigrams(BaseMetric):
    ANNOTATION_TAG = "unique_char_trigrams"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> int:
        tokens = tokenize(example)
        trigrams = list(compute_ngrams(tokens, 3))
        return len(set(trigrams))


class UniqueSubwords(BaseMetric):
    NEEDS_TOKENIZER = True
    ANNOTATION_TAG = "unique_subwords"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> int:
        tokenize_fn = kwargs.get('tokenize_fn')
        if tokenize_fn is None:
            raise ValueError("This metric requires a tokenize function.")
        tokens = tokenize_fn(example)
        return len(set(tokens))


class UniqueSubwordTrigrams(BaseMetric):
    NEEDS_TOKENIZER = True
    ANNOTATION_TAG = "unique_subword_trigrams"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> int:
        tokenize_fn = kwargs.get('tokenize_fn')
        if tokenize_fn is None:
            raise ValueError("This metric requires a tokenize function.")
        tokens = tokenize_fn(example)
        trigrams = list(compute_ngrams(tokens, n=3))
        return len(set(trigrams))


class AlphaChars(BaseMetric):
    ANNOTATION_TAG = "alpha_chars"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> int:
        return sum(1 for char in example if char.isalpha())
