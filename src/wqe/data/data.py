import json
import logging
import regex as re
import os
import importlib.resources as pkg_resources

from typing import List, Dict, Any, Union

import datasets
import fasttext
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from . import resources
from ..utils.maps import PARTITION_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

datasets.disable_caching()


class WikiLoader:

    def __init__(
            self,
            wiki_id: str,
            load_method: str = "raw",
            load_path: str = None
    ):

        """
        Loads the dataset from a local path, hub, or raw Wikimedia/Wikipedia data.

        Parameters
        ----------
        wiki_id : str
            The language id used by Wikipedia.
        load_method : str
            The method to use for loading the dataset.
            Default is `raw`, which loads the dataset from Wikimedia/Wikipedia
            using the Hugging Face `datasets` library.
            Other options are:
            - `local`: Load the dataset from a local path (must be in parquet format).
            - `hub`: Load the dataset from the Hugging Face Hub.
        load_path : str
            The path to the dataset to load (if using "local" or "hub" methods).
        """

        self.wiki_id = wiki_id
        self.regex = None
        self.lang_id_model = None

        if load_method == "local":
            logger.info(f"Loading dataset from {load_path}")
            self.data = load_dataset(load_path)
        elif load_method == "hub":
            logger.info(f"Loading dataset from hub: {load_path}/{self.wiki_id}")
            self.data = load_dataset(f"{load_path}", data_dir=f"{self.wiki_id}")
        elif load_method == "raw":
            logger.info(f"Loading raw dataset from Wikimedia/Wikipedia: {self.wiki_id}")
            self.data = load_dataset(
                "wikimedia/wikipedia",
                f"20231101.{self.wiki_id}",
                cache_dir=None
            )
        else:
            raise ValueError("Invalid import type. Please specify either 'local', 'hub', or 'raw'.")

        with pkg_resources.open_text(resources, 'wiki_mappings.json') as file:
            self.wiki_mappings = json.load(file)[self.wiki_id]

        self.n_chars = len("".join(self.data["train"]["text"]))
        self.n_docs = len(self.data["train"])
        logger.info(f"Loaded {self.n_docs} articles with {self.n_chars} characters (train).")

    def __getattr__(self, attr: str):

        return getattr(self.data, attr)

    def __getitem__(self, idx: int):

        return self.data[idx]

    def __len__(self):

        return len(self.data)

    @classmethod
    def load_split_directly(
            cls,
            wiki_id: str,
            load_method: str,
            load_path: str,
            split: str = "test"
    ) -> datasets.Dataset:

        """Loads split of previously processed wiki. Primarily used for testing models.

        Parameters
        ----------
        load_method : str
            The method to use for loading the dataset.
            Options are:
            - `local`: Load the dataset from a local path (must be in parquet format).
            - `hub`: Load the dataset from the Hugging Face Hub.
        load_path : str
            The path to the dataset to load.
        wiki_id : str
            The ID of the Wikipedia dataset to load.
        split : str, optional
            The dataset split to load (e.g., "train", "test", "validation").
            Default is "test".

        Returns
        -------
        datasets.Dataset
            The loaded dataset split.
        """

        if load_method == "local":
            logger.info(f"Loading dataset from {load_path}")
            dataset = load_dataset(load_path)
        elif load_method == "hub":
            logger.info(f"Loading dataset from hub: {load_path}/{wiki_id}")
            dataset = load_dataset(load_path, data_dir=wiki_id)
        else:
            raise ValueError("Invalid import type. Please specify either 'local' or 'hub'.")

        if split not in dataset.keys():
            raise ValueError(f"Split {split} not found in dataset. Please specify a valid split.")

        return dataset[split]

    def generate_splits(
            self,
            test_size: float = 0.1,
            shuffle: bool = True,
            seed: int = 42
    ):

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

        logger.info("Generating dataset splits...")

        self.data = self.data['train'].train_test_split(
            test_size=test_size,
            shuffle=shuffle,
            seed=seed
        )

        self.n_chars = len("".join(self.data["train"]["text"]))
        self.n_docs = len(self.data["train"])

        logger.info(f"Generated new train split with {self.n_docs} articles and {self.n_chars} characters.")

    def _make_regex(self):

        """
        Regex for filtering the dataset for all accepted unicode scripts for a language.
        """

        combined_scripts = "".join([f"\\p{{{script}}}" for script in self.wiki_mappings['scripts']])
        # script_regex = fr"[\p{{P}}\p{{S}}\d{script_to_add}]*[\d{script_to_add}]+[\p{{P}}\p{{S}}\d{script_to_add}]*\s"
        script_regex = fr"""
            [\p{{P}}\p{{S}}\d{combined_scripts}]*   # Match zero or more punctuation, symbols, digits, and scripts
            [\d{combined_scripts}]+                 # Match one or more digits and scripts
            [\p{{P}}\p{{S}}\d{combined_scripts}]*   # Match zero or more punctuation, symbols, digits, and scripts
            \s                                      # Match a whitespace character
        """
        self.regex = script_regex

    def pre_filter(
            self,
            script_regex: bool = True,
            lang_id: bool = True,
            char_cutoff: int = 15
    ):

        """
        Pre-filters the dataset using both functions and char_cutoff.

        Lang-id uses GlotLID (https://github.com/cisnlp/GlotLID)
        to predict the language of each line in the dataset.

        TODO: Make lang_id filtering faster.
        TODO: Make docst ring consistent

        Args:
            script_regex (bool): Whether to filter the dataset for accepted scripts.
            lang_id (bool): Whether to filter the dataset for the specified language.
            char_cutoff (int): The minimum number of characters required for a line to be kept.
        """
        raw_chars = self.n_chars

        if script_regex:
            self._make_regex()
            logger.info(f"Filtering documents for accepted scripts: {self.wiki_mappings['scripts']}")

        if lang_id:
            logger.info(f"Filtering documents for language: {self.wiki_mappings['alpha3']}")
            self.lang_id_model = fasttext.load_model(
                hf_hub_download(
                    repo_id="cis-lmu/glotlid",
                    filename="model.bin",
                    cache_dir=None
                )
            )

        logger.info("Removing documents shorter than 15 characters.")

        self.data = self.data.map(
            lambda example: self._pre_filter_article(example, script_regex, lang_id, char_cutoff),
            desc="Pre-filtering dataset..."
        )

        self.n_chars = len("".join(self.data["train"]["text"]))
        self.n_docs = len(self.data["train"])

        logger.info(f"Removed {raw_chars - self.n_chars} chars ({1.0 - self.n_chars / raw_chars}%).")

    def _pre_filter_article(
            self,
            article: Dict[str, Any],
            script_regex: bool = True,
            lang_id: bool = True,
            char_cutoff: int = 15
    ) -> Dict[str, Any]:

        """
        Pre-filters article using regex, lang_id, and char_cutoff.

        Args:
            article (dict): The article to pre-filter.
            script_regex (bool): Whether to filter the article for accepted scripts.
            lang_id (bool): Whether to filter the article for the specified language.
            char_cutoff (int): The minimum number of characters required for a line to be kept.

        Returns:
            dict: The pre-filtered article.
        """

        if script_regex:
            article["text"] = "".join(re.findall(self.regex, article['text']))

        if lang_id:
            article['text'] = "".join(
                [line for line in article['text'].splitlines() if len(line) > char_cutoff and
                 self.lang_id_model.predict(line)[0][0].split("_")[-2] == self.wiki_mappings['alpha3']]
            )

        return article

    def apply_partition(
            self,
            metric: Union[List[str], str],
            method: str = "balanced_chars",
            quality: bool = True,
            tokenizer: Union[str, None] = None,
            all_partitions_join_method: str = "intersection"
    ):

        """
        Updates the dataset with a partition chosen by the specified metric.
        If multiple metrics are specified, the intersection or union of the returned document indices is used.

        Args:
            method (str): The method for choosing the boundary at which to split high-quality
            and low-quality partitions. Default is 'balanced_chars', which allocates
            approximately half of the total characters to each partition.
            Also supported:
            - 'mean_cutoff': split based on the mean value of the metric
            - 'median_cutoff': split based on the median value of the metric
            - 'balanced_docs': allocates equal number of documents to each partition
            metric (List[str], str): The metric(s) to use for partitioning the dataset.
            quality (bool): Whether to return the higher-quality partition or the lower-quality partition.
            Default is True for higher-quality.
            tokenizer (str): The tokenizer to use for partitioning the dataset. Required for certain metrics.
            all_partitions_join_method (str): The method to use for joining all partitions.

        """
        raw_chars = self.n_chars
        raw_docs = self.n_docs

        if isinstance(metric, str):
            partition = PARTITION_MAP[metric](method, quality, tokenizer)
            logger.info(f"Partitioning dataset by {metric}...")

            partition_indices = partition(self.data["train"])
            self.data["train"] = self.data["train"].select(partition_indices)

        else:
            logger.info(f"Partitioning dataset by {', '.join(metric)}...")

            partition_indices = []
            for met in metric:
                partition = PARTITION_MAP[met](method, quality, tokenizer)
                partition_indices.append(partition(self.data["train"]))

            if all_partitions_join_method == "intersection":
                partition_indices = list(set.intersection(*map(set, partition_indices)))
            elif all_partitions_join_method == "union":
                partition_indices = list(set.union(*map(set, partition_indices)))
            else:
                raise ValueError("Invalid join method. Please specify either 'intersection' or 'union'.")

            self.data["train"] = self.data["train"].select(partition_indices)

        self.n_chars = len("".join(self.data["train"]["text"]))
        self.n_docs = len(self.data["train"])

        logger.info(f"Removed {raw_chars - self.n_chars} chars ({1.0 - self.n_chars / raw_chars}%).")
        logger.info(f"Removed {raw_docs - self.n_docs} docs ({1.0 - self.n_docs / raw_docs}%).")

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
