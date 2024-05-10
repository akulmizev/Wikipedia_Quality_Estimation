import json
import logging
import regex as re
import os

from importlib import resources

import datasets
import fasttext
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .partition import *
# from wqe.utils.maps import PARTITION_MAP
from utils.maps import PARTITION_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikiLoader:

    def __init__(self, config, wiki_id):

        """
        Initialize the WikiDataset object.

        Args:
            config (dict): A dictionary containing the configuration for the dataset.
        """

        self.__dict__.update(config.__dict__)
        self.wiki_id = wiki_id

        if self.load:
            if self.load.method == "local":
                logger.info(f"Loading dataset from {self.load.path}")
                self.data = load_dataset(self.load.path)
            elif self.load.method == "hub":
                logger.info(f"Loading dataset from hub: {self.load.path}/{self.wiki_id}")
                self.data = load_dataset(f"{self.load.path}", f"{self.wiki_id}")
            elif self.load.method == "raw":
                logger.info(f"Loading raw dataset from Wikimedia/Wikipedia: {self.wiki_id}")
                self.data = load_dataset(
                    "wikimedia/wikipedia",
                    f"20231101.{self.wiki_id}",
                    cache_dir=None
                )
            else:
                raise ValueError("Invalid import type. Please specify either 'local', 'hub', or 'raw'.")

        with resources.open_text("data.resources", "wiki_mappings.json") as f:
            self.wiki_mappings = json.load(f)[self.wiki_id]

        self.n_chars = len("".join(self.data["train"]["text"]))
        self.n_docs = len(self.data["train"])
        logger.info(f"Loaded {self.n_docs} articles with {self.n_chars} characters (train).")

    def __getattr__(self, attr):

        """
        Get an attribute from the dataset.

        Args:
            attr (str): The attribute to retrieve.

        Returns:
            Any: The attribute.
        """
        return getattr(self.data, attr)

    def __getitem__(self, idx):

        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.
        """
        return self.data[idx]

    def __len__(self):

        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    @classmethod
    def load_dataset_directly(cls, import_config, wiki_id, split="test"):

        if import_config.method == "local":
            logger.info(f"Loading dataset from {import_config.path}")
            dataset = load_dataset(import_config.path)
        elif import_config.method == "hub":
            logger.info(f"Loading dataset from hub: {import_config.path}/{wiki_id}")
            dataset = load_dataset(f"{import_config.path}", f"{wiki_id}")
        else:
            raise ValueError("Invalid import type. Please specify either 'local' or 'hub'.")

        if split not in dataset.keys():
            raise ValueError(f"Split {split} not found in dataset. Please specify a valid split.")

        return dataset[split]

    def generate_splits(self):

        """
        Generate the splits for the dataset.
        """

        logger.info("Generating dataset splits...")

        self.data = self.data['train'].train_test_split(
            test_size=self.split.test,
            shuffle=self.split.shuffle,
            seed=self.split.seed
        )

        self.n_chars = len("".join(self.data["train"]["text"]))
        self.n_docs = len(self.data["train"])

        logger.info(f"Generated new train split with {self.n_docs} articles and {self.n_chars} characters.")

    def _make_regex(self):

        """
        Make the regex for filtering the dataset.
        """

        script_to_add = "".join([f"\\p{{{script}}}" for script in self.wiki_mappings['scripts']])
        script_regex = fr"[\p{{P}}\p{{S}}\d{script_to_add}]*[\d{script_to_add}]+[\p{{P}}\p{{S}}\d{script_to_add}]*\s"
        self.regex = script_regex

    def _prefilter_regex(self, article):
        article["text"] = "".join(re.findall(self.regex, article['text']))

        return article

    def _prefilter_langid(self, article):
        article['text'] = "".join(
            [line for line in article['text'].splitlines() if len(line) > self.pre_filter.char_cutoff and
             self.lang_id_model.predict(line)[0][0].split("_")[-2] == self.wiki_mappings['alpha3']]
        )

        return article

    def do_pre_filter(self):

        """
        Pre-filter the dataset.

        Returns:
            datasets.Dataset: The pre-filtered dataset.
        """
        raw_chars = self.n_chars

        if self.pre_filter.script_regex:
            self._make_regex()
            logger.info(f"Filtering documents for accepted scripts: {self.wiki_mappings['scripts']}")
            self.data = self.data.map(
                self._prefilter_regex,
                desc="Pre-filtering dataset by script..."
            )

        if self.pre_filter.lang_id:
            logger.info(f"Filtering documents for language: {self.wiki_mappings['alpha3']}")
            self.lang_id_model = fasttext.load_model(
                hf_hub_download(
                    repo_id="cis-lmu/glotlid",
                    filename="model.bin",
                    cache_dir=None
                )
            )
            self.data = self.data.map(
                self._prefilter_langid,
                desc="Pre-filtering dataset by langid..."
            )

        logger.info(f"Removing articles shorter than {self.pre_filter.char_cutoff} characters.")
        self.data = self.data.filter(lambda example: len(example["text"]) > self.pre_filter.char_cutoff)

        self.n_chars = len("".join(self.data["train"]["text"]))
        self.n_docs = len(self.data["train"])

        logger.info(f"Removed {raw_chars - self.n_chars} chars ({1.0 - self.n_chars / raw_chars}%).")

    def apply_partition(self):

        """
        Update the dataset with a partition.
        """
        raw_chars = self.n_chars
        raw_docs = self.n_docs

        if self.partition.metric_type != "all":
            partition = PARTITION_MAP[self.partition.metric_type](self.partition)
            logger.info(f"Partitioning dataset by {self.partition.metric_type}...")

            partition_indices = partition(self.data["train"])
            self.data["train"] = self.data["train"].select(partition_indices)

            self.n_chars = len("".join(self.data["train"]["text"]))
            self.n_docs = len(self.data["train"])

            logger.info(f"Removed {raw_chars - self.n_chars} chars ({1.0 - self.n_chars / raw_chars}%).")
            logger.info(f"Removed {raw_docs - self.n_docs} docs ({1.0 - self.n_docs / raw_docs}%).")

        else:
            logger.info("Partitioning dataset by all metrics...")

            partition_indices = []
            for metric in [Length, UniqueTrigrams, UniqueWords, UniqueCharacters]:
                partition = metric(self.partition)
                partition_indices.append(partition(self.data["train"]))

            if self.partition.all_partitions_join_method == "intersection":
                partition_indices = list(set.intersection(*map(set, partition_indices)))
            elif self.partition.all_partitions_join_method == "union":
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
        Save the dataset to disk.

        Args:
            path (str): The path to save the dataset to.
        """

        logger.info(f"Saving dataset to: {path}")
        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")
        for split in self.data.keys():
            self.data[split].to_parquet(f"{path}/{split}.parquet")
