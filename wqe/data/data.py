import json
import logging
import regex as re
import os

from importlib import resources

import datasets
import fasttext
from datasets import load_dataset
from datasets import concatenate_datasets
from huggingface_hub import hf_hub_download
# from huggingface_hub import HfApi

from .partition import *

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class WikiDatasetFromConfig:
    def __init__(self, config):

        """
        Initialize the WikiDataset object.

        Args:
            config (dict): A dictionary containing the configuration for the dataset.
        """

        self.config = config["data"]
        self.experiment_id = config["experiment"]["id"]
        self.wiki_id = config["wiki_id"]

        if "import" in self.config:
            import_type = self.config["import"]["import_type"]

            if import_type == "local":
                path = self.config["import"]["path"]
                logging.info(f"Loading dataset from {path}")
                self.data = load_dataset(path)
            elif import_type == "hub":
                # TODO: fix this to not be redundant
                path = self.config["import"]["path"]
                logging.info(f"Loading dataset from hub: {path}/{self.wiki_id}")
                self.data = load_dataset(
                    f"{path}",
                    self.wiki_id
                )
            elif import_type == "raw":
                logging.info(f"Loading raw dataset from Wikimedia/Wikipedia: {self.wiki_id}")
                self.data = load_dataset(
                    "wikimedia/wikipedia",
                    f"20231101.{self.wiki_id}",
                    cache_dir=None
                )
            else:
                raise ValueError("Invalid import type. Please specify either 'local', 'hub', or 'raw'.")

        with resources.open_text("data.resources", "wiki_mappings.json") as f:
            self.wiki_mappings = json.load(f)[self.wiki_id]
        self.regex = None
        self.lang_id_model = None

        # TODO change this to - if loading pre_filtered data for partitions, it logs both train and test
        self.size_chars = len("".join(self.data["train"]["text"]))
        self.size_docs = len(self.data["train"])
        logging.info(f"Loaded {self.size_docs} articles with {self.size_chars} characters.")

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

    def generate_splits(self):

        """
        Generate the splits for the dataset.
        """

        logging.info("Generating dataset splits...")

        split_config = self.config["split"]

        self.data = self.data['train'].train_test_split(
            test_size=split_config["test"],
            shuffle=split_config["shuffle"],
            seed=split_config["seed"]
        )

        self.size_chars = len("".join(self.data["train"]["text"]))
        self.size_docs = len(self.data["train"])

        # self.size_chars_test = len("".join(self.data["test"]["text"]))

        logging.info(f"Generated new train split with {self.size_docs} articles and {self.size_chars} characters.")
        # logging.info(f"Generated new test split with {len(self.data['test'])} articles and {self.size_chars_test} characters.")

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
            [line for line in article['text'].splitlines() if len(line) > self.config["pre_filter"]["char_cutoff"] and
             self.lang_id_model.predict(line)[0][0].split("_")[-2] == self.wiki_mappings['alpha3']]
        )

        return article

    def pre_filter(self):

        """
        Pre-filter the dataset.

        Returns:
            datasets.Dataset: The pre-filtered dataset.
        """
        raw_size_chars = self.size_chars

        if self.config["pre_filter"]["script_regex"]:
            self._make_regex()
            logging.info(f"Filtering documents for accepted scripts: {self.wiki_mappings['scripts']}")
            self.data = self.data.map(
                self._prefilter_regex,
                desc="Pre-filtering dataset by script..."
            )
        if self.config["pre_filter"]["lang_id"]:
            logging.info(f"Filtering documents for language: {self.wiki_mappings['alpha3']}")
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

        self.size_chars = len("".join(self.data["train"]["text"]))
        self.size_docs = len(self.data["train"])

        logging.info(f"Removed {raw_size_chars - self.size_chars} chars ({1.0 - self.size_chars/raw_size_chars}%).")

    def apply_partition(self):

        """
        Update the dataset with a partition.
        """

        partition_map = {
            "length": Length,
            "unique_subwords": UniqueSubwords,
            "unique_subword_trigrams": UniqueSubwordTrigrams,
            "unique_trigrams": UniqueTrigrams,
            "unique_words": UniqueWords,
            "unique_characters": UniqueCharacters,
            "unique_character_trigrams": UniqueCharacterTrigrams,
            "alpha_chars": AlphaChars
        }

        partition_metric = self.config["partition"]["partition_metric"]

        raw_size_chars = self.size_chars
        raw_size_docs = self.size_docs

        if partition_metric != "all":
            partition = partition_map[partition_metric](self.config)
            logging.info(f"Partitioning dataset by {partition_metric}...")

            self.data["train"] = partition(self.data["train"])

            self.size_chars = len("".join(self.data["train"]["text"]))
            self.size_docs = len(self.data["train"])

            logging.info(f"Removed {raw_size_chars - self.size_chars} chars ({1.0 - self.size_chars/raw_size_chars}%).")
            logging.info(f"Removed {raw_size_docs - self.size_docs} docs ({1.0 - self.size_docs/raw_size_docs}%).")
        else:
            logging.info("Partitioning dataset by all metrics...")

            high_quality = []
            for metric in [Length, UniqueTrigrams, UniqueWords, UniqueCharacters]:
                partition = metric(self.config)
                high_quality.append(partition(self.data["train"]))

            data_ids = concatenate_datasets(high_quality)["id"]
            if self.config["partition"]["quality"]:
                self.data = self.data.filter(lambda x: x["id"] in data_ids)
            else:
                self.data = self.data.filter(lambda x: x["id"] not in data_ids)

            self.size_chars = len("".join(self.data["train"]["text"]))
            self.size_docs = len(self.data["train"])

            logging.info(f"Removed {raw_size_chars - self.size_chars} chars ({1.0 - self.size_chars / raw_size_chars}%).")
            logging.info(f"Removed {raw_size_docs - self.size_docs} docs ({1.0 - self.size_docs / raw_size_docs}%).")

    def save(self):

        """
        Save the dataset to disk.
        """

        path = self.config["export"]["path"]
        export_type = self.config["export"]["export_type"]

        if export_type == "hub":
            logging.info(f"Pushing dataset to hub: {path}/{self.experiment_id}")
            self.data.push_to_hub(
                repo_id=f"{path}/{self.experiment_id}",
                data_dir=f"{self.wiki_id}",
                config_name=f'{self.wiki_id}',
                private=True
            )
        elif export_type == "local":
            logging.info(f"Saving dataset to: {path}/{self.experiment_id}/{self.wiki_id}")
            if not os.path.exists(f"{path}/{self.experiment_id}"):
                os.makedirs(f"{path}/{self.experiment_id}")
            self.data.save_to_disk(
                f"{path}/{self.experiment_id}/{self.wiki_id}"
            )
        else:
            logging.info("Invalid export type. Please specify either 'hub' or 'local'.")
            logging.info("Skipping export.")
