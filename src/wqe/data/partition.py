import logging
import numpy as np

from typing import Any, Dict, List, Optional, Union

import datasets
import multiprocessing as mp

from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import vstack, csr_matrix
from transformers import PreTrainedTokenizerFast

from ..utils.maps import PARTITION_MAP

logger = logging.getLogger(__name__)


# class TfIdfSelfSimilarity(Partition):
#
#     """
#     Partition a dataset based on the number of unique subwords trigrams in its articles.
#     Pre-trained tokenization must be provided.
#
#     Parameters
#     ----------
#     tokenizer : str
#         A model string used for loading a pre-trained tokenization, e.g. "bert-base-multilingual-cased".
#     **kwargs
#         Additional keyword arguments passed to the Partition base class.
#
#     Methods
#     -------
#     metric(example)
#         Compute the number of unique subword trigrams in the given example.
#     """
#
#     def __init__(self, tokenizer: str, **kwargs):
#
#         if not tokenizer:
#             assert ValueError("Pass a tokenization for this metric.")
#
#         super().__init__(tokenizer=tokenizer, **kwargs)
#
#         self.vectorizer = HashingVectorizer(
#             alternate_sign=False,  # All weights positive
#             norm='l2',  # Normalize by L2 norm
#             n_features=self.tokenizer.vocab_size,
#             tokenizer=self.tokenizer.tokenize,
#             lowercase=False,
#             dtype=np.float32
#         )
#
#         self.higher_is_better = False
#
#     def process_batch(self, batch):
#
#         return self.vectorizer.transform(batch).tocsr()
#
#     def __call__(
#             self,
#             dataset: datasets.Dataset,
#             batch_size: int = 1000
#     ) -> List[int]:
#
#         batches = [batch for batch in self.batch_iterator(dataset["text"], batch_size)]
#
#         with mp.Pool(mp.cpu_count()) as pool:
#             tfidf_matrix = vstack(pool.map(self.process_batch, batches))
#
#         avg_similarities = self.sparse_cosine_similarity_matrix_chunked(tfidf_matrix)
#         metric_per_doc = np.abs(avg_similarities - np.median(avg_similarities))
#
#         return self._select_partition(dataset, metric_per_doc)
#
#     def metric(self, example: str) -> int:
#         tokens = self.tokenizer.tokenize(example)
#         trigrams = list(ngrams(tokens, 3))
#         return len(set(trigrams))
#
#     @staticmethod
#     def batch_iterator(dataset, batch_size=1000):
#         for i in range(0, len(dataset), batch_size):
#             yield dataset[i:i + batch_size]
#
#     @staticmethod
#     def sparse_cosine_similarity_matrix_chunked(matrix, batch_size=1000):
#
#         n_samples = matrix.shape[0]
#
#         average_similarities = np.zeros(n_samples)
#
#         for i in range(0, n_samples, batch_size):
#             end = min(i + batch_size, n_samples)
#             batch = matrix[i:end]
#
#             similarities = batch.dot(matrix.T)
#
#             row_sums = similarities.sum(axis=1).A1 - 1  # Subtract 1 to exclude self-similarity
#             average_similarities[i:end] = row_sums / (n_samples - 1)
#
#             print(f"Processed rows {i} to {end}")
#
#         return average_similarities


class Partition:

    def __init__(
            self,
            metrics: Union[str, List[str]],
            method: str = "mean_cutoff",
            quality: bool = True,
            join_method: Optional[str] = None,
            thresholds: Dict[str, int] = None,
            tokenizer: str = None
    ):

        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.method = method
        self.quality = quality
        self.join_method = join_method
        self.thresholds = thresholds if thresholds else None
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer) if tokenizer else None

    def __call__(
        self,
        dataset: datasets.Dataset
    ) -> datasets.Dataset:

        dataset = dataset.map(
            self.apply_metrics,
            fn_kwargs={"metrics": self.metrics, "tokenizer": self.tokenizer},
            num_proc=mp.cpu_count()
        )

        if self.thresholds:
            for metric, threshold in self.thresholds.items():
                if metric not in self.metrics:
                    raise ValueError(f"Threshold specified for non-existent metric: {metric}.")

                if threshold == "auto":
                    self.thresholds[metric] = self.get_auto_threshold(dataset[metric], self.quality)

            dataset = dataset.filter(
                self.apply_thresholds,
                fn_kwargs={"thresholds": self.thresholds},
            )

        if self.join_method:
            dataset = self.join_partitions(dataset)

        return dataset

    def select_indices(
        self,
        dataset: datasets.Dataset,
        metric: str,
    ) -> List[int]:

        metric_per_doc = np.array(dataset[metric])
        higher_is_better = PARTITION_MAP[metric].HIGHER_IS_BETTER

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
            sorted_indices = np.argsort(metric_per_doc)
            text_lengths = np.array([len(text) for text in dataset["text"]])
            total_chars = text_lengths.sum()
            char_budget = total_chars // 2

            cumulative_chars = np.cumsum(text_lengths[sorted_indices])
            partition_point = np.searchsorted(cumulative_chars, char_budget, side='right')

            partition_1 = sorted_indices[:partition_point]
            partition_2 = sorted_indices[partition_point:]
        else:
            raise ValueError("Partition method not recognized.")

        if (higher_is_better and self.quality) or (not higher_is_better and not self.quality):
            return partition_2
        else:
            return partition_1

    def join_partitions(self, dataset) -> List[int]:

        partition_indices = [self.select_indices(dataset, metric) for metric in self.metrics]

        if self.join_method == "intersection":
            partition_indices = list(set.intersection(*map(set, partition_indices)))
        elif self.join_method == "union":
            partition_indices = list(set.union(*map(set, partition_indices)))
        else:
            raise ValueError("Invalid join method. Please specify either 'intersection' or 'union'.")

        dataset = dataset.select(partition_indices)

        return dataset

    @staticmethod
    def apply_metrics(
        example: Dict[str, Any],
        metrics: List[str],
        tokenizer: PreTrainedTokenizerFast = None
    ) -> Dict[str, Any]:

        for metric_name in metrics:
            metric = PARTITION_MAP[metric_name]
            if metric.NEEDS_TOKENIZER:
                if not tokenizer:
                    raise ValueError(f"{metric_name} requires a tokenizer.")
                example[metric_name] = metric.calculate(example["text"], tokenize_fn=tokenizer.tokenize)
            else:
                example[metric_name] = metric.calculate(example["text"])

        return example

    @staticmethod
    def apply_thresholds(
        example: Dict[str, Any],
        thresholds: Dict[str, int]
    ) -> datasets.Dataset:

        for metric, threshold in thresholds.items():

            higher_is_better = PARTITION_MAP[metric].HIGHER_IS_BETTER

            if higher_is_better:
                if example[metric] < threshold:
                    return False
            else:
                if example[metric] > threshold:
                    return False

        return True

    @staticmethod
    def get_auto_threshold(
        values: List[Union[int, float]],
        quality: bool
    ) -> int:

        metric_values = np.array(values)

        if quality:
            return np.quantile(metric_values, 0.75)
        else:
            return np.quantile(metric_values, 0.25)
