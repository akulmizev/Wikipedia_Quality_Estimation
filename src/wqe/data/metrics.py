import unicodedata

from typing import Any

from .utils import *

ALL_PUNCTUATION = get_all_punctuation()


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


class FracAllCapsWords(BaseMetric):
    ANNOTATION_TAG = "frac_all_caps_words"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> float:
        words = tokenize(example)
        all_caps_words = [word for word in words if word.isupper()]
        return len(all_caps_words) / len(words)


class FracLinesEndEllipsis(BaseMetric):
    ANNOTATION_TAG = "frac_lines_end_ellipsis"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> float:
        lines = example.split("\n")
        ellipsis_lines = [line for line in lines if line.endswith("...")]
        return len(ellipsis_lines) / len(lines)


class FracNoAlphaWords(BaseMetric):
    ANNOTATION_TAG = "frac_no_alpha_words"

    @classmethod
    def is_script(cls, token: str) -> bool:
        return all(
            unicodedata.category(char).startswith('L')
            for char in token
            if not unicodedata.category(char).startswith('M')
        )

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> float:
        words = tokenize(example)
        no_alpha_words = [word for word in words if not any(cls.is_script(char) for char in word)]
        return len(no_alpha_words) / len(words)


class MeanWordLength(BaseMetric):
    ANNOTATION_TAG = "doc_mean_word_length"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> float:
        words = tokenize(example)
        return sum(len(word) for word in words) / len(words)


class FracUniqueWords(BaseMetric):
    ANNOTATION_TAG = "frac_unique_words"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> float:
        words = tokenize(example)
        return len(set(words)) / len(words)


class FracUniqueChars(BaseMetric):
    ANNOTATION_TAG = "frac_unique_chars"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> float:
        return len(set(example)) / len(example)


class UnigramEntropy(BaseMetric):
    ANNOTATION_TAG = "unigram_entropy"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> float:
        tokens = tokenize(example)
        token_set = set(tokens)
        token_counts = {token: tokens.count(token) for token in token_set}
        total_tokens = len(tokens)
        return -sum((count / total_tokens) * (count / total_tokens) for count in token_counts.values())


class TrigramEntropy(BaseMetric):
    ANNOTATION_TAG = "trigram_entropy"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> float:
        tokens = tokenize(example)
        trigrams = list(compute_ngrams(tokens, 3))
        trigram_set = set(trigrams)
        trigram_counts = {trigram: trigrams.count(trigram) for trigram in trigram_set}
        total_trigrams = len(trigrams)
        return -sum((count / total_trigrams) * (count / total_trigrams) for count in trigram_counts.values())


class LinesEndWithPunctuation(BaseMetric):
    ANNOTATION_TAG = "lines_end_with_punctuation"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> bool:
        lines = example.split("\n")
        return all(line[-1] in ALL_PUNCTUATION for line in lines)


class NumLines(BaseMetric):
    ANNOTATION_TAG = "num_lines"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> int:
        return len(example.split("\n"))


class NumWordsPerLine(BaseMetric):
    ANNOTATION_TAG = "num_words_per_line"

    @classmethod
    def calculate(cls, example: str, **kwargs: Any) -> float:
        lines = example.split("\n")
        return sum(len(tokenize(line)) for line in lines) / len(lines)
