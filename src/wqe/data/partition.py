import numpy as np

from typing import List, Dict, Any, Union

import datasets

from nltk.util import ngrams
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


class Partition:

    """
    Base class for partitioning a dataset based on a specified metric.

    Parameters
    ----------
    method : str, optional
        The method for choosing the boundary at which to split high-quality and low-quality partitions.
        Default is 'balanced_chars', which allocates approximately half of the total characters to each partition.
        Also supported:
            - 'mean_cutoff': split based on the mean value of the metric
            - 'median_cutoff': split based on the median value of the metric
            - 'balanced_docs': allocates equal number of documents to each partition
    quality : bool, optional
        Whether to return the higher-quality partition or the lower-quality partition.
        Default is True for higher-quality.
    tokenizer : FastTokenizerFromConfig, optional
        A tokenizer to use for tokenizing the dataset. Required for certain metrics.
    **kwargs
        Additional keyword arguments.

    Methods
    -------
    __call__(dataset)
        Split the dataset into a partition based on the specified metric.
    metric(example)
        Compute the metric for a given example. Must be implemented in a subclass.
    output_stats(**kwargs)
        Output statistics for the partitioning. Must be implemented in a subclass.
    """

    def __init__(
        self,
        method: str = "balanced_chars",
        quality: bool = True,
        tokenizer: PreTrainedTokenizerFast = None,
        **kwargs
    ):

        self.method = method
        self.quality = quality
        self.higher_is_better = False
        if tokenizer:
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer)

    def __call__(self, dataset: datasets.Dataset) -> List[int]:

        """
        Split the dataset into a partition based on the specified metric.

        Parameters
        ----------
        dataset : Dataset
            The dataset object to partition. Assumes access to a 'text' field.

        Returns
        -------
        partition : List[int]
            Indices of the partition specified by `quality`.
        """

        metric_per_doc = [self.metric(item) for item in dataset["text"]]

        if self.method == "mean_cutoff":
            mean_cutoff = np.mean(metric_per_doc)  # + np.std(metric_per_doc)
            partition_1 = np.where(metric_per_doc < mean_cutoff)[0]
            partition_2 = np.where(metric_per_doc >= mean_cutoff)[0]
        elif self.method == "median_cutoff":
            median_cutoff = np.median(metric_per_doc)
            partition_1 = np.where(metric_per_doc < median_cutoff)[0]
            partition_2 = np.where(metric_per_doc >= median_cutoff)[0]
        elif self.method == "balanced_docs":
            half_point = len(dataset) // 2
            partition_1 = np.argsort(metric_per_doc)[:half_point]
            partition_2 = np.argsort(metric_per_doc)[half_point:]
        elif self.method == "balanced_chars":
            sorted_indices = np.argsort(metric_per_doc).tolist()
            char_budget = len("".join(dataset["text"])) // 2
            char_counter = 0
            partition_1 = []
            while char_counter < char_budget:
                char_counter += len(dataset[sorted_indices[0]]["text"])
                partition_1.append(sorted_indices.pop(0))
            partition_2 = sorted_indices
        else:
            raise ValueError("Partition method not recognized.")

        if (self.higher_is_better and self.quality) or (not self.higher_is_better and not self.quality):
            return partition_2
        else:
            return partition_1

    def metric(self, example: str) -> int:
        raise NotImplementedError("Metric not implemented. Please use a subclass.")

    def output_stats(self, **kwargs) -> None:
        raise NotImplementedError("Stats not implemented. Please use a subclass.")


class Length(Partition):

    """
    Partition a dataset based on the length of its articles, in characters.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to the Partition base class.

    Methods
    -------
    metric(example)
        Compute the length of the given example in characters.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.higher_is_better = True

    def metric(self, example: str) -> int:
        return len(example)


class UniqueSubwords(Partition):

    """
    Partition a dataset based on the number of unique subwords in its articles.
    Pre-trained tokenizer must be provided.

    Parameters
    ----------
    tokenizer : str
        A model string used for loading a pre-trained tokenizer, e.g. "bert-base-multilingual-cased".
    **kwargs
        Additional keyword arguments passed to the Partition base class.

    Methods
    -------
    metric(example)
        Compute the number of unique subwords in the given example.
    """

    def __init__(self, tokenizer: str, **kwargs):
        if not tokenizer:
            raise ValueError("Pass a tokenizer for this metric.")
        super().__init__(**kwargs)
        self.higher_is_better = True

    def metric(self, example: str) -> int:
        tokens = self.tokenizer.tokenize(example)
        return len(set(tokens))


class UniqueSubwordTrigrams(Partition):

    """
    Partition a dataset based on the number of unique subwords trigrams in its articles.
    Pre-trained tokenizer must be provided.

    Parameters
    ----------
    tokenizer : str
        A model string used for loading a pre-trained tokenizer, e.g. "bert-base-multilingual-cased".
    **kwargs
        Additional keyword arguments passed to the Partition base class.

    Methods
    -------
    metric(example)
        Compute the number of unique subword trigrams in the given example.
    """

    def __init__(self, tokenizer: str, **kwargs):
        if not tokenizer:
            raise ValueError("Pass a tokenizer for this metric.")
        super().__init__(**kwargs)
        self.higher_is_better = True

    def metric(self, example: str) -> int:
        tokens = self.tokenizer.tokenize(example)
        trigrams = list(ngrams(tokens, 3))
        return len(set(trigrams))


class UniqueTrigrams(Partition):

    """
    Partition a dataset based on the number of unique trigrams in its articles.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to the Partition base class.

    Methods
    -------
    metric(example)
        Compute the number of unique trigrams in the given example.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.higher_is_better = True

    def metric(self, example: str) -> int:

        words = [word[0] for word in Whitespace().pre_tokenize_str(example)]
        trigrams = list(ngrams(words, 3))
        return len(set(trigrams))


class UniqueWords(Partition):

    """
    Partition a dataset based on the number of unique unigrams in its articles.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to the Partition base class.

    Methods
    -------
    metric(example)
        Compute the number of unique unigrams in the given example.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.higher_is_better = True

    def metric(self, example: str) -> int:
        words = [word[0] for word in Whitespace().pre_tokenize_str(example)]
        return len(set(words))


class UniqueCharacters(Partition):

    """
    Partition a dataset based on the number of unique characters in its articles.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to the Partition base class.

    Methods
    -------
    metric(example)
        Compute the number of unique characters in the given example.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.higher_is_better = True

    def metric(self, example: str) -> int:
        return len(set(example))


class UniqueCharacterTrigrams(Partition):

    """
    Partition a dataset based on the number of unique character trigrams in its articles.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to the Partition base class.

    Methods
    -------
    metric(example)
        Compute the number of unique character trigrams in the given example.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.higher_is_better = True

    def metric(self, example: str) -> int:
        trigrams = list(ngrams(example, 3))
        return len(set(trigrams))


class AlphaChars(Partition):

    """
    Partition a dataset based on the number of alphabetic in its articles.
    Assumes `alphabetic` refers to the characters in the English alphabet, i.e [a-zA-Z].

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to the Partition base class.

    Methods
    -------
    metric(example)
        Compute the number of alphabetic characters in the given example.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.higher_is_better = False

    def metric(self, example: str) -> int:
        return sum([1 for char in example if char.isalpha()])
