import logging
import numpy as np

from typing import Any, Dict, List, Optional, Union

import datasets
import multiprocessing as mp
from kneed import KneeLocator

from numpy.random import random_sample
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import vstack, csr_matrix
from scipy import stats, optimize
from scipy.stats import gaussian_kde
from transformers import PreTrainedTokenizerFast

from .utils import tokenize
from ..utils.maps import METRIC_MAP

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
            **kwargs
    ):

        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.method = method
        self.quality = quality
        self.join_method = join_method
        self.thresholds = thresholds

        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer)
            elif isinstance(tokenizer, PreTrainedTokenizerFast):
                self.tokenizer = tokenizer
            else:
                raise ValueError("Invalid tokenizer type. Pass a model string or a PreTrainedTokenizerFast object.")
        else:
            self.tokenizer = None

        self.model = kwargs.get("model", None)

    def __call__(
        self,
        dataset: datasets.Dataset,
    ) -> datasets.Dataset:

        logger.info(f"Partitioning dataset by {', '.join(self.metrics)}...")

        if "perplexity" in self.metrics:
            METRIC_MAP["perplexity"].init_metrics(dataset, model=self.model)

        dataset = dataset.map(
            self.apply_metrics,
            fn_kwargs={"metrics": self.metrics, "tokenizer": self.tokenizer},
            num_proc=mp.cpu_count(),
            desc="Calculating metrics"
        )

        if self.thresholds:
            for metric, threshold in self.thresholds.items():
                if metric not in self.metrics:
                    raise ValueError(f"Threshold specified for non-existent metric: {metric}.")

                higher_is_better = METRIC_MAP[metric].HIGHER_IS_BETTER
                if threshold == "auto":
                    self.thresholds[metric] = self.get_auto_threshold(dataset[metric], higher_is_better)
                    if higher_is_better:
                        n_docs_removed = len(np.where(dataset[metric] < self.thresholds[metric])[0])
                        logger.info(
                            f"Auto threshold for {metric} set at: <{self.thresholds[metric]:.2f} "
                            f"({n_docs_removed} docs removed)"
                        )
                    else:
                        n_docs_removed = len(np.where(dataset[metric] > self.thresholds[metric])[0])
                        logger.info(
                            f"Auto threshold for {metric} set at: >{self.thresholds[metric]:.2f} "
                            f"({n_docs_removed} matching docs)"
                        )

            dataset = dataset.filter(
                self.apply_thresholds,
                fn_kwargs={"thresholds": self.thresholds},
                num_proc=mp.cpu_count(),
                desc="Applying thresholds"
            )

        if self.join_method:
            dataset = self.join_partitions(dataset)
        else:
            indices = self.select_indices(dataset, self.metrics[0])
            dataset = dataset.select(indices)

        dataset = dataset.remove_columns(self.metrics)

        return dataset

    def select_indices(
        self,
        dataset: datasets.Dataset,
        metric: str,
    ) -> List[int]:

        metric_per_doc = np.array(dataset[metric])
        higher_is_better = METRIC_MAP[metric].HIGHER_IS_BETTER

        if self.method == "mean_cutoff":
            mean_cutoff = np.mean(metric_per_doc)  # + np.std(metric_per_doc)
            partition_1 = np.where(metric_per_doc < mean_cutoff)[0]
            partition_2 = np.where(metric_per_doc >= mean_cutoff)[0]
        elif self.method == "median_cutoff":
            median_cutoff = np.median(metric_per_doc)
            partition_1 = np.where(metric_per_doc < median_cutoff)[0]
            partition_2 = np.where(metric_per_doc >= median_cutoff)[0]
        elif self.method == "balanced_docs":
            partition_point = len(dataset) // 2
            partition_1 = np.argsort(metric_per_doc)[:partition_point]
            partition_2 = np.argsort(metric_per_doc)[partition_point:]
        elif self.method == "balanced_chars":
            sorted_indices = np.argsort(metric_per_doc)
            text_lengths = np.array([len(text) for text in dataset["text"]])
            total_chars = text_lengths.sum()
            char_budget = total_chars // 2

            cumulative_chars = np.cumsum(text_lengths[sorted_indices])
            partition_point = np.searchsorted(cumulative_chars, char_budget, side='right')

            partition_1 = sorted_indices[:partition_point]
            partition_2 = sorted_indices[partition_point:]
        elif self.method == "elbow":
            sorted_indices = np.argsort(metric_per_doc)[::-1]
            scores = np.sort(metric_per_doc)[::-1]
            # scores = (scores-np.min(scores))/(np.max(scores)-np.min(scores)) * 100
            n_values = np.arange(0, len(scores))
            knee = KneeLocator(n_values, scores, curve='convex', direction='decreasing', interp_method="polynomial")
            logger.info(f"Knee point found at {knee.knee}")

            partition_2 = sorted_indices[:knee.knee]
            partition_1 = sorted_indices[knee.knee:]
        else:
            raise ValueError("Partition method not recognized.")

        if (higher_is_better and self.quality) or (not higher_is_better and not self.quality):
            return partition_2
        else:
            return partition_1

    def join_partitions(self, dataset) -> List[int]:
        if len(self.metrics) == 1:
            logger.warning("Only one metric passed, but join_method specified.")

        if self.join_method == "intersection":
            partition_indices = [self.select_indices(dataset, metric) for metric in self.metrics]
            partition_indices = list(set.intersection(*map(set, partition_indices)))
            dataset = dataset.select(partition_indices)
            return dataset
        elif self.join_method == "union":
            partition_indices = [self.select_indices(dataset, metric) for metric in self.metrics]
            partition_indices = list(set.union(*map(set, partition_indices)))
            dataset = dataset.select(partition_indices)
            return dataset
        elif self.join_method == "scores":
            assert self.method == 'elbow', "Currently, `scores` join_method can only be used in conjunction with the `elbow` method."
            all_scores = np.zeros(len(dataset))
            for metric in self.metrics:
                scores = (dataset[metric] - np.min(dataset[metric])) / (np.max(dataset[metric]) - np.min(dataset[metric])) * 100
                all_scores += scores
            sorted_scores = np.sort(all_scores)[::-1]
        #     variances = [np.var(dataset['all_scores'][:i]) for i in range(1, len(dataset))]
            n_values = np.arange(0, len(sorted_scores)) #range starts from 1 if using variances
            knee = KneeLocator(n_values, sorted_scores, curve='convex', direction='decreasing', interp_method="polynomial")
            logger.info(f"Knee point found at {knee.knee}")
            dataset = dataset.add_column("all_scores", all_scores)
            sorted_indices = np.argsort(dataset["all_scores"])[::-1]
            if self.quality:
                dataset = dataset.select(sorted_indices[:knee.knee]).remove_columns("all_scores")
                return dataset
            else:
                dataset = dataset.select(sorted_indices[knee.knee:]).remove_columns("all_scores")
                return dataset
        else:
            raise ValueError("Invalid join method. Please specify either 'intersection', 'union' or 'scores'.")


    @staticmethod
    def apply_metrics(
        example: Dict[str, Any],
        metrics: List[str],
        tokenizer: PreTrainedTokenizerFast = None
    ) -> Dict[str, Any]:

        if tokenizer:
            example["tokens"] = tokenizer.tokenize(example["text"])
        else:
            example["tokens"] = tokenize(example["text"])

        for metric_name in metrics:
            metric = METRIC_MAP[metric_name]
            example = metric.calculate(example)

        return example

    @staticmethod
    def apply_thresholds(
        example: Dict[str, Any],
        thresholds: Dict[str, int]
    ) -> datasets.Dataset:

        for metric, threshold in thresholds.items():

            higher_is_better = METRIC_MAP[metric].HIGHER_IS_BETTER

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
        higher_is_better: bool
    ) -> int:

        n_docs = int(len(values) * 0.05)
        sorted_values = np.array(sorted(values)[:n_docs]) if higher_is_better else np.array(sorted(values)[n_docs:])

        if len(set(sorted_values)) == 1:
            return sorted_values[0]

        sampled_values = np.random.choice(values, n_docs, replace=False)

        value_kde = gaussian_kde(sorted_values)
        sampled_kde = gaussian_kde(sampled_values)

        if higher_is_better:
            metric_values = np.linspace(min(sorted_values), max(sampled_values), 1000)
            threshold = metric_values[np.argmax(value_kde(metric_values) - sampled_kde(metric_values))]
        else:
            metric_values = np.linspace(min(sampled_values), max(sorted_values), 1000)
            threshold = metric_values[np.argmin(value_kde(metric_values) - sampled_kde(metric_values))]

        return threshold

    @staticmethod
    def get_auto_threshold_optimized(
            values: List[Union[int, float]],
            higher_is_better: bool
    ) -> float:
        values = np.array(values)
        n_docs = max(int(len(values) * 0.05), 2)  # Ensure at least 2 docs

        if higher_is_better:
            sorted_values = np.partition(values, n_docs)[:n_docs]
        else:
            sorted_values = np.partition(values, -n_docs)[-n_docs:]

        if len(set(sorted_values)) == 1:
            return sorted_values[0]

        sampled_values = values[np.random.permutation(len(values))[:n_docs]]

        value_kde = stats.gaussian_kde(sorted_values)
        sampled_kde = stats.gaussian_kde(sampled_values)

        if higher_is_better:
            objective = lambda x: sampled_kde(x) - value_kde(x)
            bounds = (min(sorted_values), max(sampled_values))
        else:
            objective = lambda x: value_kde(x) - sampled_kde(x)
            bounds = (min(sampled_values), max(sorted_values))

        result = optimize.minimize_scalar(objective, bounds=bounds, method='bounded')
        return result.x
