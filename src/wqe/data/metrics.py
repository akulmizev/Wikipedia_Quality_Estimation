import math
import numpy as np
import torch

from collections import Counter
from datasets import Dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from typing import Any, Dict
from tqdm import tqdm

from .utils import *

ALL_PUNCTUATION = get_all_punctuation()


class BaseMetric:
    NEEDS_TOKENIZER: bool = False
    HIGHER_IS_BETTER: bool = True
    ANNOTATION_TAG: str = "default"

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> int:
        raise NotImplementedError("Metric not implemented. Please use a subclass.")


class LengthCharacters(BaseMetric):
    ANNOTATION_TAG = "length_chars"

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        example[cls.ANNOTATION_TAG] = len(example["text"])
        return example


class LengthWords(BaseMetric):
    ANNOTATION_TAG = "length_words"

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        example[cls.ANNOTATION_TAG] = len(example["tokens"])
        return example


class UniqueWords(BaseMetric):
    ANNOTATION_TAG = "unique_words"

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        example[cls.ANNOTATION_TAG] = len(set(example["tokens"]))
        return example


class UniqueTrigrams(BaseMetric):
    ANNOTATION_TAG = "unique_trigrams"

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        trigrams = list(compute_ngrams(example["tokens"], 3))
        example[cls.ANNOTATION_TAG] = len(set(trigrams))
        return example


class UniqueCharacters(BaseMetric):
    ANNOTATION_TAG = "unique_chars"

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        example[cls.ANNOTATION_TAG] = len(set(example["text"]))
        return example


class UniqueCharacterTrigrams(BaseMetric):
    ANNOTATION_TAG = "unique_char_trigrams"

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        trigrams = list(compute_ngrams(example["text"], 3))
        example[cls.ANNOTATION_TAG] = len(set(trigrams))
        return example


class AlphaChars(BaseMetric):
    ANNOTATION_TAG = "alpha_chars"

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        example[cls.ANNOTATION_TAG] = sum(1 for char in example["text"] if char.isalpha())
        return example


class FracAllCapsWords(BaseMetric):
    ANNOTATION_TAG = "frac_all_caps_words"
    HIGHER_IS_BETTER = True

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        all_caps_words = [word for word in example["tokens"] if word.isupper()]
        example[cls.ANNOTATION_TAG] = len(all_caps_words) / len(example["tokens"])
        return example


class FracLinesEndEllipsis(BaseMetric):
    ANNOTATION_TAG = "frac_lines_end_ellipsis"
    HIGHER_IS_BETTER = False

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        lines = example["text"].split("\n")
        ellipsis_lines = [line for line in lines if line.endswith("...")]
        example[cls.ANNOTATION_TAG] = len(ellipsis_lines) / len(lines)
        return example


class FracNoScriptWords(BaseMetric):
    ANNOTATION_TAG = "frac_no_script_words"
    HIGHER_IS_BETTER = True

    @classmethod
    def is_script(cls, char: str) -> bool:
        char_category = unicodedata.category(char)
        return char_category.startswith('L') or char_category.startswith('M')

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        no_alpha_words = [word for word in example["tokens"] if not any(cls.is_script(char) for char in word)]
        example[cls.ANNOTATION_TAG] = len(no_alpha_words) / len(example["tokens"])
        return example


class MeanWordLength(BaseMetric):
    ANNOTATION_TAG = "doc_mean_word_length"
    HIGHER_IS_BETTER = True

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        example[cls.ANNOTATION_TAG] = sum(len(word) for word in example["tokens"]) / len(example["tokens"])
        return example


class FracUniqueWords(BaseMetric):
    ANNOTATION_TAG = "frac_unique_words"
    HIGHER_IS_BETTER = True

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        example[cls.ANNOTATION_TAG] = len(set(example["tokens"])) / len(example["tokens"])
        return example


class FracUniqueCharacters(BaseMetric):
    ANNOTATION_TAG = "frac_unique_chars"
    HIGHER_IS_BETTER = True

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        example[cls.ANNOTATION_TAG] = len(set(example["text"])) / len(example["text"])
        return example


class FracUniqueTrigrams(BaseMetric):
    ANNOTATION_TAG = "frac_unique_trigrams"
    HIGHER_IS_BETTER = True

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        trigrams = list(compute_ngrams(example["tokens"], 3))
        if len(trigrams) == 0:
            example[cls.ANNOTATION_TAG] = 0.0
        else:
            example[cls.ANNOTATION_TAG] = len(set(trigrams)) / len(trigrams)
        return example


class UnigramEntropy(BaseMetric):
    ANNOTATION_TAG = "unigram_entropy"
    HIGHER_IS_BETTER = True

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        tokens = example["tokens"]
        token_counts = Counter(tokens)  # Efficient counting
        total_tokens = len(tokens)

        # Calculate token probabilities
        token_probs = {token: count / total_tokens for token, count in token_counts.items()}

        # Calculate entropy
        entropy = -sum(prob * math.log2(prob) for prob in token_probs.values())
        example[cls.ANNOTATION_TAG] = entropy
        return example


class TrigramEntropy(BaseMetric):
    ANNOTATION_TAG = "trigram_entropy"
    HIGHER_IS_BETTER = True

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        trigrams = list(compute_ngrams(example["tokens"], 3))
        trigram_counts = Counter(trigrams)  # Efficient counting
        total_trigrams = len(trigrams)

        # Calculate trigram probabilities
        trigram_probs = {trigram: count / total_trigrams for trigram, count in trigram_counts.items()}

        # Calculate entropy
        entropy = -sum(prob * math.log2(prob) for prob in trigram_probs.values())
        example[cls.ANNOTATION_TAG] = entropy
        return example


class CharacterEntropy(BaseMetric):
    ANNOTATION_TAG = "char_entropy"
    HIGHER_IS_BETTER = True

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        chars = list(example["text"])
        char_counts = Counter(chars)  # Efficient counting
        total_chars = len(chars)

        # Calculate character probabilities
        char_probs = {char: count / total_chars for char, count in char_counts.items()}

        # Calculate entropy
        entropy = -sum(prob * math.log2(prob) for prob in char_probs.values())
        example[cls.ANNOTATION_TAG] = entropy
        return example


class LinesEndWithPunctuation(BaseMetric):
    ANNOTATION_TAG = "lines_end_with_punctuation"
    HIGHER_IS_BETTER = True

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        lines = example["text"].split("\n")
        score = sum(1 for line in lines if line.strip().endswith(tuple(ALL_PUNCTUATION))) / len(lines)
        example[cls.ANNOTATION_TAG] = score
        return example


class NumLines(BaseMetric):
    ANNOTATION_TAG = "num_lines"
    HIGHER_IS_BETTER = True

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        example[cls.ANNOTATION_TAG] = len(example["text"].split("\n"))
        return example


class NumWordsPerLine(BaseMetric):
    ANNOTATION_TAG = "num_words_per_line"
    HIGHER_IS_BETTER = True

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        lines = example["text"].split("\n")
        example[cls.ANNOTATION_TAG] = sum(len(line.split()) for line in lines) / len(lines)
        return example


class ModelBasedMetric(BaseMetric):
    model = None
    metrics = None

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if not cls.metrics:
            raise ValueError("Metrics must be calculated before calling the metric.")
        example[cls.ANNOTATION_TAG] = cls.metrics[example["id"]]
        return example

    @classmethod
    def init_metrics(cls, examples: Dataset, **kwargs: Any):
        raise NotImplementedError("This method must be implemented in a subclass.")


class Perplexity(ModelBasedMetric):
    HIGHER_IS_BETTER = False
    ANNOTATION_TAG = "perplexity"
    metrics = None

    @classmethod
    def calculate(cls, example: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if not cls.metrics:
            raise ValueError("Perplexities must be calculated via `init_metrics` before calling the metric.")
        example[cls.ANNOTATION_TAG] = cls.metrics[example["id"]]
        return example

    @classmethod
    def init_metrics(cls, examples: Dataset, **kwargs: Any):
        model_name = kwargs.get("model", None)
        if model_name is None:
            raise ValueError("A model must be provided for the perplexity metric.")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')
        cls.metrics = {}
        for i in tqdm(range(0, len(examples)), desc="Calculating perplexities..."):
            example = examples[i]
            batch = [example["text"]]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            with torch.no_grad():
                try:
                    outputs = model(**inputs, labels=inputs["input_ids"])
                except RuntimeError:
                    pass
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                cls.metrics[example["id"]] = perplexity
        ppls = np.array(list(cls.metrics.values()))
        normed_ppls = np.abs(ppls - np.median(ppls))
        cls.metrics = {id_: ppl for id_, ppl in zip(cls.metrics.keys(), normed_ppls)}
        del model
        del tokenizer
        torch.cuda.empty_cache()
