import json
import logging
import os
import importlib.resources as pkg_resources

from dataclasses import dataclass
from typing import Dict, List, Union

import datasets

from datasets import load_dataset, DatasetDict
from datasets.exceptions import DatasetNotFoundError
from numpy.random import choice
from transformers import PreTrainedTokenizerFast

from . import resources
from .processing import PreFilter, Deduplicate, Threshold, Partition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

datasets.disable_caching()


@dataclass
class WikiID:

    """
    Class for handling Wikipedia language IDs.

    Parameters
    ----------
    id : str
        The language id used by Wikipedia. For example, "as" for Assamese.
        Can be found in the `Wiki` column here:
        https://meta.wikimedia.org/wiki/List_of_Wikipedias.

    Attributes
    ----------
    id : str
        The language id used by Wikipedia.
    scripts : list
        The unicode scripts accepted for the language.
    alpha3 : str
        The ISO 639-3 code for the language.
    language : str
        The language name.
    """

    id: str

    def __str__(self):
        return self.id

    def __post_init__(self):
        with pkg_resources.open_text(resources, 'wiki_mappings.json') as file:
            wiki_mappings = json.load(file)
        try:
            self.scripts = wiki_mappings[self.id]['scripts']
            self.alpha3 = wiki_mappings[self.id]['alpha3']
            self.language = wiki_mappings[self.id]['language']
        except KeyError:
            raise ValueError(f"Invalid Wiki ID: {self.id}. Please choose a valid Wiki ID."
                             f"Use `WikiLoader.show_available_languages()` to see available wikis.")


class WikiLoader:

    """
    Class for loading and preprocessing Wikipedia datasets.
    Ideally, this should inherit from `datasets.Dataset` in the future,
    but this was not trivial to implement given the `arrow` functionality
    assumed there.

    Parameters
    ----------
    wiki_id : str
        The language id used by Wikipedia. For example, "as" for Assamese.
        Can be found in the `Wiki` column here:
        https://meta.wikimedia.org/wiki/List_of_Wikipedias.

    Attributes
    ----------
    wiki : WikiID
        The WikiID instance for the specified language.
    data : datasets.DatasetDict
        The dataset loaded from `wikipedia/wikimedia` via `datasets.load_dataset`.
    n_chars : int
        The total number of characters in the dataset.
    n_docs : int
        The total number of documents in the dataset.
    """

    def __init__(
            self,
            wiki_id: str
    ):

        """
        Initializes a WikiLoader instance.

        Parameters
        ----------
        wiki_id : str
            The language id used by Wikipedia. For example, "as" for Assamese.
            Can be found in the `Wiki` column here:
            https://meta.wikimedia.org/wiki/List_of_Wikipedias.
        """

        self.wiki = WikiID(wiki_id)
        self.data = None
        self.n_chars = 0
        self.n_docs = 0
        self.columns = ["id", "url", "title", "text"]

    def __getattr__(self, attr: str):
        return getattr(self.data, attr)

    def __getitem__(self, split: str):
        return self.data[split]

    def __repr__(self):
        return f"WikiLoader(wiki_id='{self.wiki.id}', n_docs={self.n_docs}, n_chars={self.n_chars})"

    def __str__(self):
        return f"WikiLoader for {self.wiki.language} ({self.wiki.alpha3})\n" \
               f"Wiki ID: {self.wiki.id}\n" \
               f"Articles: {self.n_docs}\n" \
               f"Characters: {self.n_chars}\n" \
               f"Scripts: {', '.join(self.wiki.scripts)}"

    def get_doc(
            self,
            idx: int = None
    ):

        """
        Returns the first document in the dataset. Useful for testing.

        Parameters
        ----------
        idx : int
            The index of the document to return. If not specified, a random document is returned.

        Returns
        -------
        str
            The text of the document.
        """

        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."

        idx = choice(range(len(self.data["train"]["text"]))) if idx is None else idx

        return self.data["train"]["text"][idx]

    @staticmethod
    def show_available_wikis():

        """
        Prints a table of available languages to load from Wikimedia/Wikipedia,
        including their alpha3 codes and scripts.
        """

        with pkg_resources.open_text(resources, 'wiki_mappings.json') as file:
            wiki_mappings = json.load(file)

        print(f"{'Wiki ID':<15}{'Language':<40}{'639-3':<10}{'Scripts':<30}")
        print("-" * 88)
        for wiki_id, wiki in sorted(wiki_mappings.items(), key=lambda x: x[1]['language']):
            print(f"{wiki_id:<15}{wiki['language']:<40}{wiki['alpha3']:<10}{', '.join(wiki['scripts']):<30}")

    def update_counts(self):
        self.n_chars = len("".join(self.data["train"]["text"]))
        self.n_docs = len(self.data["train"])

    @classmethod
    def from_dataset(
        cls,
        dataset: datasets.Dataset,
        wiki_id: str
    ):

        """
        Initializes a WikiLoader instance from a datasets.Dataset.

        Parameters
        ----------
        dataset : datasets.Dataset
            The dataset to load the dataset from.
        wiki_id : str
            The language id used by Wikipedia. For example, "as" for Assamese.
            Can be found in the `Wiki` column here:
            https://meta.wikimedia.org/wiki/List_of_Wikipedias.

        Returns
        -------
        WikiLoader
            The initialized WikiLoader instance.
        """

        instance = cls(wiki_id)
        instance.data = DatasetDict({"train": dataset})
        instance.n_chars = len("".join(instance.data["train"]["text"]))
        instance.n_docs = len(instance.data["train"])

        logger.info(f"Loaded {instance.n_docs} articles with {instance.n_chars} characters (train).")

        return instance

    def load_dataset(
        self,
        load_path: str = None,
        split: str = None,
        dump_date: str = "20231101"
    ) -> 'WikiLoader':

        """
        Loads the dataset from a local path, hub, or raw Wikimedia/Wikipedia dataset.
        If split (e.g `test`) is specified, only that split is loaded.
        Otherwise, all splits are loaded.

        Parameters
        ----------
        load_path : str
            The path to the dataset to load locally or from the huggingface hub.
            Will raise either `DatasetNotFoundError` if the dataset is not found in either location.
            If loading locally, assumes a directory structure of `load_path/{wiki_id}`.
            Loads a raw dataset from Wikimedia/Wikipedia if not specified.
        split : str, optional
            The dataset split to load (e.g., "train", "test", "validation").
            Loads all splits if not specified.
        dump_date : str, optional
            The dump date for the Wikimedia/Wikipedia dataset.
            Default is "20231101".
        """

        if load_path:
            try:
                if os.path.exists(load_path):
                    self.data = load_dataset(load_path)
                else:
                    self.data = load_dataset(load_path, self.wiki.id)
            except (DatasetNotFoundError, FileNotFoundError, ValueError):
                raise DatasetNotFoundError(f"Could not find dataset at {load_path}/{self.wiki.id}.")
        else:
            self.data = load_dataset(
                "wikimedia/wikipedia",
                f"{dump_date}.{self.wiki.id}",
                cache_dir=None
            )

        if split:
            if split not in self.data.keys():
                raise ValueError(f"Split {split} not found in dataset. Please specify a valid split.")
            else:
                self.data = DatasetDict({split: self.data[split]})
        else:
            split = "train"

        self.n_chars = len("".join(self.data[split]["text"]))
        self.n_docs = len(self.data[split])

        logger.info(f"Loaded {self.n_docs} articles with {self.n_chars} characters ({split}). Wiki: {self.wiki.id}")

        return self

    def generate_splits(
        self,
        test_size: float = 0.1,
        shuffle: bool = True,
        seed: int = 42
    ) -> 'WikiLoader':

        """
        Generates train and test splits for the specified wiki.

        Parameters
        ----------
        test_size : float, optional
            The size of the test split. Default is 0.1.
        shuffle : bool, optional
            Whether to shuffle the dataset before splitting. Default is True.
        seed : int, optional
            The random seed to use for shuffling. Default is 42.
        """

        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."

        logger.info("Generating dataset splits...")

        self.data = self.data['train'].train_test_split(
            test_size=test_size,
            shuffle=shuffle,
            seed=seed
        )

        self.update_counts()

        logger.info(f"Generated new train split with {self.n_docs} articles and {self.n_chars} characters.")

        return self

    def pre_filter(
        self,
        script_regex: bool = False,
        lang_id: bool = False,
        apply_c4_filter: bool = False,
        urls_to_remove: List[str] = None,
        warn_percent: float = 0.0,
        **kwargs
    ):

        """
        Pre-filters the dataset using the following functions:

        - `script_regex`: Removes lines from the dataset that do not contain
        any of the accepted scripts for the language (e.g. Cyrillic for English).

        - `lang_id`: Removes lines from the dataset that are not identified as
        belonging to the specified language. This is done via the GlotLID model.
        CAUTION: This is very slow and should be used sparingly, as it is not
        guaranteed to be accurate for lower-resourced languages.

        - `apply_c4_filter`: Removes lines from the dataset that do not meet the
        Common Crawl C4 dataset criteria.

        - `urls_to_remove`: Removes articles with specified URLs from the dataset.

        This method first makes a full pass through the dataset in order to apply the
        `script_regex` and `lang_id` filters, and compute hashes for articles.
        It then makes a second pass through the dataset to remove articles according
        to the `char_cutoff`, `deduplicate_exact_match`, and `deduplicate_min_hash` filters.

        The filters can be applied simultaneously...

        ```
        from wqe import WikiLoader
        loader = WikiLoader("ha")
        loader.pre_filter(
            script_regex=True,
            lang_id=True,
            apply_c4_filter=True
        )
        ```

        ...or successively:

        ```
        from wqe import WikiLoader
        loader = WikiLoader("ha")
        loader.pre_filter(script_regex=True)
        loader.pre_filter(lang_id=True)
        loader.pre_filter(apply_c4_filter=True)
        ```

        It is recommended to use the `num_proc` parameter to speed up filtering
        for large datasets. However, the `lang_id` filter is not supported for
        multiprocessing, as the GlotLID model is not thread-safe.

        Parameters
        ----------
        script_regex : bool
            Whether to filter the dataset for accepted scripts.
            Default is False.
        lang_id : bool
            Whether to filter the dataset for the specified language.
            Default is False
        apply_c4_filter : bool
            Whether to filter the dataset for the Common Crawl C4 dataset criteria.
            Default is False.
        urls_to_remove : list
            The list of URLs to remove from the dataset.
            Useful for buggy articles such as https://xh.wikipedia.org/wiki/Phi.
        warn_percent : float
            Warn when the percentage of removed characters exceeds this value.
        """

        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."
        assert "train" in self.data.keys(), "Function requires a train split."

        scripts_to_keep = self.wiki.scripts if script_regex else None
        langs_to_keep = [self.wiki.alpha3] if lang_id else None

        prefilter = PreFilter(
            scripts_to_keep=scripts_to_keep,
            langs_to_keep=langs_to_keep,
            apply_c4_filter=apply_c4_filter
        )

        self.data["train"] = prefilter(
            self.data["train"],
            urls_to_remove=urls_to_remove,
            warn_percent=warn_percent,
            **kwargs
        )

        self.update_counts()

        return self

    def deduplicate(
        self,
        exact_match: bool = False,
        min_hash: bool = False,
        jaccard_threshold: float = 0.85,
        n_shingles: int = 3,
        tokenizer: str = None,
        **kwargs
    ):

        """
        Deduplicates the dataset using the following methods:

        - `deduplicate_exact_match`: Removes duplicate articles by hashing
        the text of each article and removing exact match duplicates.

        - `deduplicate_min_hash`: Removes duplicate articles by computing
        the Jaccard similarity between article unigrams using MinHash-LSH,
        and filtering based on the specified threshold. Can be used in conjunction
        with a trained tokenization, if provided. Otherwise, will lowercase
        and split on whitespace.
        -------
        Parameters

        exact_match : bool
            Whether to deduplicate the dataset by exact match.
            Default is False.
        min_hash : bool
            Whether to deduplicate the dataset by MinHash-LSH.
            Default is False.
        jaccard_threshold : float
            The Jaccard (set) similarity threshold for MinHash-LSH.
            Default is 0.85.
        n_shingles : int
            The number of shingles to use for MinHash-LSH.
            Default is 3.
        tokenizer : str
            Tokenizer to use for MinHash-LSH.
            If not provided, will split on whitespace
        """

        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."
        assert "train" in self.data.keys(), "Function requires a train split."

        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer) if tokenizer else None

        deduplicate = Deduplicate(
            exact_match=exact_match,
            min_hash=min_hash,
            jaccard_threshold=jaccard_threshold,
            n_shingles=n_shingles
        )

        self.data["train"] = deduplicate(self.data["train"], tokenizer=tokenizer, **kwargs)

        self.update_counts()

        return self

    def apply_threshold(
        self,
        thresholds: Dict[str, Union[int, float, str]],
        tokenizer: str = None,
        **kwargs
    ):

        """
        Filters the dataset based on the specified thresholds for each metric.
        If a metric is not specified in the thresholds, no threshold is applied.
        All implemented metrics can be found in the `wqe.data.metrics` module.

        Thresholds can also be estimated automatically from the metric distribution
        by specifying 'auto'. Implementation can be found in
        `wqe.data.processing.Threshold._get_auto_threshold()`.

        Parameters
        ----------
        thresholds : dict
            The thresholds for filtering by each metric, e.g. `length: 100`.
            If not specified, no threshold is applied.
        tokenizer : str
            Tokenizer to use for various metrics, e.g. length in words.
            If not provided, will split on whitespace.
        """

        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."
        assert "train" in self.data.keys(), "Function requires a train split."

        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer) if tokenizer else None

        threshold = Threshold(thresholds, **kwargs)
        self.data["train"] = threshold(self.data["train"], tokenizer=tokenizer, **kwargs)

        self.update_counts()

        return self

    def apply_partition(
        self,
        split_method: str,
        metrics: Union[List[str], str],
        quality: bool = True,
        join_partitions_by: str = None,
        tokenizer: str = None,
        **kwargs
    ):

        """
        Updates the dataset with a partition chosen by the specified metric.

        If multiple metrics are specified, a join method must be specified,
        where either the intersection or union of the returned document indices
        is used, or each document is scored based on the given metrics and partitions
        are created based on the scores.

        Parameters
        ----------
        split_method : str
            The method for choosing the boundary at which to split high-quality
            and low-quality partitions. Default is 'balanced_chars', which allocates
            approximately half of the dataset's total characters to each partition.
            Also supported:
            - 'mean_cutoff': split based on the mean value of the metric
            - 'median_cutoff': split based on the median value of the metric
            - 'balanced_docs': allocates equal number of documents to each partition
            - 'elbow': uses the elbow method to determine the optimal cutoff for distribution
        metrics : list of str or str
            The metric(s) to use for partitioning the dataset.
        quality : bool
            Whether to return the higher-quality partition or the lower-quality partition.
            Default is True for higher-quality.
        join_partitions_by : str
            If a list of metrics is specified, specifies how to join them.
            Set operations are performed on the dataset indices returned by each metric.
            Choice between 'intersection' and 'union'.
        tokenizer : str
            Tokenizer to use for various metrics, e.g. length in words.
            If not provided, will split on whitespace.
        kwargs : dict
            Additional keyword arguments for the partitioning method.
        """

        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."
        assert "train" in self.data.keys(), "Function requires a train split."

        # if "test" in self.data.keys():
        #     self.data["train"] = datasets.concatenate_datasets([self.data["train"], self.data["test"]])

        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer) if tokenizer else None

        partition = Partition(
            split_method=split_method,
            metrics=metrics,
            quality=quality,
            join_partitions_by=join_partitions_by,
            **kwargs
        )
        self.data["train"] = partition(self.data["train"], tokenizer=tokenizer, **kwargs)

        self.update_counts()

        return self

    def save(self, path):

        """
        Saves the dataset to disk.

        Args:
            path (str): The path to save the dataset to.
        """

        logger.info(f"Saving dataset to: {path}")
        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")

        for split in self.data.keys():
            self.data[split].to_parquet(f"{path}/{split}.parquet")
