import logging
import numpy as np
import regex as re
import pickle

from typing import Any, Dict, List, Optional, Pattern, Union

import fasttext
import multiprocessing as mp
import tqdm

from datasets import Dataset, DatasetDict
from datasketch import MinHash, LeanMinHash, MinHashLSH
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import insecure_hashlib
from kneed import KneeLocator
from numpy import ndarray
from scipy.stats import gaussian_kde
from transformers import PreTrainedTokenizerFast

from .utils import c4_filter, compute_ngrams, tokenize, measure_deletion
from ..utils.maps import METRIC_MAP
from ..utils.stats import normalize

logger = logging.getLogger(__name__)


class PreFilter:

    def __init__(
        self,
        scripts_to_keep: List[str] = None,
        langs_to_keep: List[str] = None,
        apply_c4_filter: bool = False
    ):
        assert scripts_to_keep or langs_to_keep or apply_c4_filter, \
            "At least one filter must be specified for pre-filtering."

        self.patterns = {
            "whitespace": re.compile(r"\s+"),
            "tokens": re.compile(r"[^\w\s]"),
            "terminal_punct": re.compile(r"[\p{Po}\p{Pf}]")
        }

        if scripts_to_keep:
            self.scripts_to_keep = scripts_to_keep
            self._make_regex(self)
        else:
            self.scripts_to_keep = None

        if langs_to_keep:
            self.lang_id = True
            self.langs_to_keep = langs_to_keep
            self.lang_id_model = fasttext.load_model(
                hf_hub_download(
                    repo_id="cis-lmu/glotlid",
                    filename="model.bin",
                    cache_dir=None
                )
            )
        else:
            self.langs_to_keep = None
            self.lang_id_model = None

        self.apply_c4_filter = apply_c4_filter

    @measure_deletion
    def __call__(
        self,
        dataset: Union[Dataset, DatasetDict],
        urls_to_remove: List[str] = None,
        warn_percent: float = 0.0,
        num_proc: int = mp.cpu_count()
    ) -> Dataset:

        if self.scripts_to_keep:
            logger.info(f"Filtering documents for accepted scripts: {', '.join(self.scripts_to_keep)}")
        if self.langs_to_keep:
            logger.info(f"Filtering documents for accepted languages: {', '.join(self.langs_to_keep)}")
        if self.apply_c4_filter:
            logger.info("Applying C4 filter.")

        dataset = dataset.map(
            function=self._pre_filter_doc,
            fn_kwargs={
                "patterns": self.patterns,
                "langs_to_keep": self.langs_to_keep,
                "apply_c4_filter": self.apply_c4_filter,
                "model": self.lang_id_model,
                "urls_to_remove": urls_to_remove,
                "warn_percent": warn_percent
            },
            num_proc=num_proc if not self.langs_to_keep else 1,
            desc="Pre-filtering dataset.",
        )

        dataset = dataset.filter(
            lambda x: x["text"].strip(),
            num_proc=num_proc,
            desc="Removing empty documents."
        )

        return dataset

    @staticmethod
    def _pre_filter_doc(
        doc: Dict[str, Any],
        patterns: Dict[str, Pattern],
        model: fasttext.FastText = None,
        langs_to_keep: List[str] = None,
        apply_c4_filter: bool = False,
        urls_to_remove: List[str] = None,
        warn_percent: float = 0.0
    ) -> Dict[str, Any]:

        """
        Pre-filters article using regex and lang-id - in that order.
        Also accepts a list of URLs to remove from the dataset, if necessary.
        Primarily used for calling in the `datasets.Dataset.map` function.

        Parameters
        ----------
        doc : dict
            The article to pre-filter.
        patterns : dict
            The regex patterns to use for filtering.
        langs_to_keep : list
            The list of languages to keep in the article.
        apply_c4_filter : bool
            Whether to apply the Common Crawl C4 filter.
        model : fasttext.FastText
            The GlotLID model for predicting the language of a line.
        apply_c4_filter : bool
            Whether to apply the Common Crawl C4 filter.
        urls_to_remove : list
            The list of URLs to remove from the article.
        warn_percent : float
            Warn when the percentage of removed characters exceeds this value.

        Returns
        -------
        dict
            The pre-filtered document.
        """

        if urls_to_remove:
            assert "url" in doc, "URLs to remove specified, but document does not have a `url` field."
            if doc["url"] in urls_to_remove:
                doc["text"] = ""
                return doc

        article_length = len(doc["text"])
        lines = doc["text"].splitlines()
        filtered_lines = []

        for line in lines:
            if not line.strip():
                continue
            else:
                line = line.strip()
            if "scripts" in patterns:
                line = "".join(re.findall(patterns["scripts"], line))
                if "cleanup" in patterns:
                    line = re.sub(patterns["cleanup"], lambda x: " " if x.group(0) else "", line)
                    line = re.sub(patterns["whitespace"], " ", line)
                if not line.strip():
                    continue
            if langs_to_keep:
                assert model is not None, "Language ID model must be specified for language filtering."
                pred_lang = model.predict(line)[0][0].split("_")[-2]
                if pred_lang not in langs_to_keep:
                    continue
            if apply_c4_filter:
                if not c4_filter(line, patterns):
                    continue

            filtered_lines.append(line)

        doc["text"] = "\n".join(filtered_lines)

        if warn_percent > 0.0:
            removed_chars = article_length - len(doc["text"])
            if removed_chars / article_length > warn_percent:
                logger.warning(f"Removed {removed_chars} characters from article: {doc['id']}")

        return doc

    @staticmethod
    def _make_regex(self):

        """
        Makes regex for filtering the dataset for all accepted Unicode scripts for a language.
        Patterns are compiled here to avoid runtime compilation.
        """

        scripts = "".join([fr"\p{{{script}}}" for script in self.scripts_to_keep])
        accepted_characters = r"\p{M}\p{P}\p{S}\p{N}\p{Z}\p{C}"
        script_regex = fr"[{accepted_characters}]*{scripts}+[{accepted_characters}]*"
        self.patterns["scripts"] = re.compile(script_regex)
        brackets = r"[\(\[\{][^\p{L}]+[\)\]\}]"
        cleanup_pattern = fr"^(?!.*{scripts}).*$|(?<=\S)\s+(?=\.$)|{brackets}|^\s*\S+\s*$"
        self.patterns["cleanup"] = re.compile(cleanup_pattern)


class Deduplicate:

    N_PERM = 64

    def __init__(
        self,
        exact_match: bool = False,
        min_hash: bool = False,
        jaccard_threshold: float = 0.85,
        n_shingles: int = 1
    ):
        assert exact_match or min_hash, \
            "Either exact_match or min_hash must be specified for deduplication."

        self.exact_match = exact_match
        self.min_hash = min_hash
        self.jaccard_threshold = jaccard_threshold
        self.n_shingles = n_shingles

    @measure_deletion
    def __call__(
        self,
        dataset: Union[Dataset, DatasetDict],
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        num_proc: int = mp.cpu_count(),
    ) -> Dataset:

        columns_to_keep = dataset.column_names

        logger.info("Deduplicating dataset.")
        if not tokenizer and self.min_hash:
            logger.warning("No tokenizer specified. Splitting on whitespace for minhash.")

        dataset = dataset.map(
            self._deduplicate_doc,
            fn_kwargs={
                "exact_match": self.exact_match,
                "min_hash": self.min_hash,
                "tokenizer": tokenizer,
                "n_shingles": self.n_shingles
            },
            num_proc=num_proc,
            desc="Calculating hashes."
        )

        unique_hashes = set(dataset.unique("hash")) if self.exact_match else None
        lsh = MinHashLSH(threshold=self.jaccard_threshold, num_perm=Deduplicate.N_PERM) if self.min_hash else None

        docs_to_keep = {}

        for doc in tqdm.tqdm(dataset, desc="Querying hashing indices for duplicates."):
            keep_doc = True
            if self.exact_match:
                if doc["hash"] not in unique_hashes:
                    keep_doc = False
                else:
                    unique_hashes.remove(doc["hash"])

            if self.min_hash:
                minhash = pickle.loads(doc["min_hash"])
                if lsh.query(minhash):
                    keep_doc = False
                else:
                    lsh.insert(doc["id"], minhash)

            docs_to_keep[doc["id"]] = keep_doc

        dataset = dataset.filter(
            lambda x: docs_to_keep[x["id"]],
            num_proc=num_proc,
            desc="Removing duplicates."
        )

        return dataset.select_columns(columns_to_keep)

    @staticmethod
    def _deduplicate_doc(
        doc: Dict[str, Any],
        exact_match: bool = False,
        min_hash: bool = False,
        tokenizer: PreTrainedTokenizerFast = None,
        n_shingles: int = 1,
    ) -> Dict[str, Any]:

        """
        Deduplicates article using exact_match and min-hash deduplication.
        Primarily used for calling in the `datasets.Dataset.map` function.

        Parameters
        ----------
        doc : dict
            The article to deduplicate.
        exact_match : bool
            Whether to deduplicate using exact match.
        min_hash : bool
            Whether to deduplicate using min-hash.
        tokenizer : PreTrainedTokenizerFast
            The tokenizer to use for min-hash when computing shingles.
        n_shingles : int
            The number of shingles to use for min-hash deduplication.
            Default is 1.

        Returns
        -------
        dict
            The deduplicated article.
        """

        if "tokens" not in doc.keys():
            doc["tokens"] = tokenize(doc["text"]) if not tokenizer else tokenizer.tokenize(doc["text"])

        if exact_match:
            doc["hash"] = insecure_hashlib.md5("".join(doc["tokens"]).encode("utf-8")).hexdigest()

        if min_hash:
            shingles = compute_ngrams(doc["tokens"], n_shingles)
            minhash = MinHash(num_perm=Deduplicate.N_PERM)
            for shingle in shingles:
                minhash.update(shingle.encode("utf-8"))
            doc["min_hash"] = pickle.dumps(LeanMinHash(minhash))

        return doc


class Threshold:

    def __init__(
        self,
        thresholds: Dict[str, Union[int, float, str]],
        **kwargs
    ):
        self.thresholds = thresholds
        self.metrics = list(thresholds.keys())
        # Only needed for perplexity
        self.model = kwargs.get("model", None)

    @measure_deletion
    def __call__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerFast = None,
        keep_columns: bool = False,
        num_proc: int = mp.cpu_count(),
        **kwargs
    ) -> Dataset:

        metrics_to_calculate = [metric for metric in self.metrics if metric not in dataset.column_names]

        logger.info(f"Thresholding dataset by {', '.join(metrics_to_calculate)}...")

        if "perplexity" in metrics_to_calculate:
            METRIC_MAP["perplexity"].init_metrics(dataset, model=self.model)

        if len(metrics_to_calculate) > 0:
            dataset = dataset.map(
                self._apply_metrics,
                fn_kwargs={
                    "metrics": metrics_to_calculate,
                    "tokenizer": tokenizer
                },
                num_proc=num_proc,
                desc="Calculating metrics."
            )
        else:
            logger.info("Metrics already calculacted. Skipping calculation.")

        for metric, threshold in self.thresholds.items():
            higher_is_better = METRIC_MAP[metric].HIGHER_IS_BETTER
            if threshold == "auto":
                self.thresholds[metric] = self._get_auto_threshold(dataset[metric], higher_is_better)
                if higher_is_better:
                    n_docs_removed = len(np.where(np.array(dataset[metric]) < self.thresholds[metric])[0])
                    logger.info(
                        f"Auto threshold for {metric} set at: <{self.thresholds[metric]:.2f} "
                        f"({n_docs_removed} docs removed)"
                    )
                else:
                    n_docs_removed = len(np.where(np.array(dataset[metric]) > self.thresholds[metric])[0])
                    logger.info(
                        f"Auto threshold for {metric} set at: >{self.thresholds[metric]:.2f} "
                        f"({n_docs_removed} matching docs)"
                    )

        dataset = dataset.filter(
            self._apply_thresholds,
            fn_kwargs={"thresholds": self.thresholds},
            num_proc=num_proc,
            desc="Applying thresholds"
        )

        if not keep_columns:
            columns_to_keep = list(set(dataset.column_names) - set(self.metrics + ["tokens"]))
            dataset = dataset.select_columns(columns_to_keep)

        return dataset

    @staticmethod
    def _apply_metrics(
        doc: Dict[str, Any],
        metrics: List[str],
        tokenizer: PreTrainedTokenizerFast = None
    ) -> Dict[str, Any]:

        if "tokens" not in doc.keys():
            doc["tokens"] = tokenize(doc["text"]) if not tokenizer else tokenizer.tokenize(doc["text"])

        for metric_name in metrics:
            metric = METRIC_MAP[metric_name]
            doc = metric.calculate(doc)

        return doc

    @staticmethod
    def _apply_thresholds(
        example: Dict[str, Any],
        thresholds: Dict[str, int]
    ) -> bool:

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
    def _get_auto_threshold(
        values: List[Union[int, float]],
        higher_is_better: bool
    ) -> float:

        if len(values) == 0:
            raise ValueError("The 'values' list cannot be empty.")

        n_docs = int(len(values) * 0.05)
        sorted_values = np.array(sorted(values)[:n_docs]) if higher_is_better else np.array(sorted(values)[n_docs:])

        # Check if variance is too low
        if np.var(sorted_values) < 1e-10:
            # Return a default threshold based on whether higher is better
            return float(sorted_values[0] if higher_is_better else sorted_values[-1])

        if np.sum(sorted_values == 0) > 0.9 * len(sorted_values):
            return 0.0

        sampled_values = np.random.choice(values, n_docs, replace=False)
        value_kde = gaussian_kde(sorted_values)
        sampled_kde = gaussian_kde(sampled_values)

        if higher_is_better:
            metric_values = np.linspace(min(sorted_values), max(sampled_values), n_docs)
            threshold = metric_values[np.argmax(value_kde(metric_values) - sampled_kde(metric_values))]
        else:
            metric_values = np.linspace(min(sampled_values), max(sorted_values), n_docs)
            threshold = metric_values[np.argmin(value_kde(metric_values) - sampled_kde(metric_values))]

        return threshold


class Partition:

    def __init__(
        self,
        metrics: Union[str, List[str]],
        split_method: str,
        quality: bool = True,
        join_partitions_by: Optional[str] = None,
        **kwargs
    ):

        self.split_method = split_method
        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.quality = quality
        self.join_partitions_by = join_partitions_by
        # Only needed for perplexity
        self.model = kwargs.get("model", None)

    @measure_deletion
    def __call__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerFast = None,
        keep_columns: bool = False,
        num_proc: int = mp.cpu_count(),
        **kwargs
    ) -> Dataset:

        metrics_to_calculate = [metric for metric in self.metrics if metric not in dataset.column_names]

        logger.info(f"Partition splitting method set to '{self.split_method}'.")
        logger.info(f"Partitioning dataset by {', '.join(self.metrics)}...")

        if "perplexity" in self.metrics:
            METRIC_MAP["perplexity"].init_metrics(dataset, model=self.model)

        if len(metrics_to_calculate) > 0:
            dataset = dataset.map(
                self.apply_metrics,
                fn_kwargs={"metrics": metrics_to_calculate, "tokenizer": tokenizer},
                num_proc=mp.cpu_count(),
                desc="Calculating metrics."
            )
        else:
            logger.info("Metrics already calculacted. Skipping calculation.")

        if self.join_partitions_by:
            dataset = self.join_partitions(dataset)
        else:
            if len(self.metrics) > 1:
                logger.warning(f"Multiple metrics passed, but no join_method specified."
                               " Combining and scoring metrics instead.")
                dataset = self.combine_and_score_metrics(dataset)
                self.metrics = ["combined_metrics"]
            indices = self.select_indices(dataset, self.metrics[0])
            dataset = dataset.select(indices)
            if "combined_metrics" in dataset.column_names:
                dataset = dataset.remove_columns(["combined_metrics"])

        if not keep_columns:
            columns_to_keep = list(set(dataset.column_names) - set(self.metrics + ["tokens"]))
            dataset = dataset.select_columns(columns_to_keep)

        return dataset

    def select_indices(
        self,
        dataset: Dataset,
        metric: str,
    ) -> ndarray[Any]:

        metric_per_doc = np.array(dataset[metric])
        if metric == "combined_metrics":
            higher_is_better = True
        else:
            higher_is_better = METRIC_MAP[metric].HIGHER_IS_BETTER

        if self.split_method == "mean_cutoff":
            mean_cutoff = np.mean(metric_per_doc)  # + np.std(metric_per_doc)
            partition_1 = np.where(metric_per_doc < mean_cutoff)[0]
            partition_2 = np.where(metric_per_doc >= mean_cutoff)[0]
        elif self.split_method == "median_cutoff":
            median_cutoff = np.median(metric_per_doc)
            partition_1 = np.where(metric_per_doc < median_cutoff)[0]
            partition_2 = np.where(metric_per_doc >= median_cutoff)[0]
        elif self.split_method == "balanced_docs":
            partition_point = len(dataset) // 2
            partition_1 = np.argsort(metric_per_doc)[:partition_point]
            partition_2 = np.argsort(metric_per_doc)[partition_point:]
        elif self.split_method == "balanced_chars":
            sorted_indices = np.argsort(metric_per_doc)
            text_lengths = np.array([len(text) for text in dataset["text"]])
            total_chars = text_lengths.sum()
            char_budget = total_chars // 2
            cumulative_chars = np.cumsum(text_lengths[sorted_indices])
            partition_point = np.searchsorted(cumulative_chars, char_budget, side='right')
            partition_1 = sorted_indices[:partition_point]
            partition_2 = sorted_indices[partition_point:]
        elif self.split_method == "elbow":
            sorted_indices = np.argsort(metric_per_doc)[::-1]
            scores = np.array(metric_per_doc[sorted_indices])
            # scores = (scores-np.min(scores))/(np.max(scores)-np.min(scores)) * 100
            n_values = np.arange(0, len(scores))
            knee = KneeLocator(
                n_values,
                scores,
                curve='convex',
                direction='decreasing',
                interp_method="polynomial"
            )
            logger.info(f"Knee point found at {knee.knee}")
            partition_1 = sorted_indices[:knee.knee]
            partition_2 = sorted_indices[knee.knee:]
        else:
            raise ValueError(f"Partition method not recognized: {self.split_method}.")

        if (higher_is_better and self.quality) or (not higher_is_better and not self.quality):
            return partition_2
        else:
            return partition_1

    def join_partitions(self, dataset) -> List[int]:

        if len(self.metrics) == 1:
            logger.warning("Only one metric passed, but join_method specified.")

        if self.join_partitions_by == "intersection":
            partition_indices = [self.select_indices(dataset, metric) for metric in self.metrics]
            partition_indices = list(set.intersection(*map(set, partition_indices)))
            dataset = dataset.select(partition_indices)
            return dataset
        elif self.join_partitions_by == "union":
            partition_indices = [self.select_indices(dataset, metric) for metric in self.metrics]
            partition_indices = list(set.union(*map(set, partition_indices)))
            dataset = dataset.select(partition_indices)
            return dataset
        else:
            raise ValueError("Invalid join method. Please specify either 'intersection', 'union' or 'scores'.")

    def combine_and_score_metrics(
        self,
        dataset: Dataset
    ):

        combined_metrics = np.zeros(len(dataset))
        for metric in self.metrics:
            scores = np.array(dataset[metric])
            norm_scores = normalize(scores)
            combined_metrics += norm_scores
        columns_to_keep = list(set(dataset.column_names) - set(self.metrics))
        dataset = dataset.select_columns(columns_to_keep)
        dataset = dataset.add_column("combined_metrics", combined_metrics)
        return dataset

    @staticmethod
    def apply_metrics(
        example: Dict[str, Any],
        metrics: List[str],
        tokenizer: PreTrainedTokenizerFast = None
    ) -> Dict[str, Any]:

        if "tokens" not in example.keys():
            example["tokens"] = tokenize(example["text"]) if not tokenizer else tokenizer.tokenize(example["text"])

        for metric_name in metrics:
            metric = METRIC_MAP[metric_name]
            example = metric.calculate(example)

        return example
