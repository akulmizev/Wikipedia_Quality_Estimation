import json
import regex as re
import os

from importlib import resources

import datasets
import fasttext
from huggingface_hub import hf_hub_download


class WikiDatasetFromConfig:
    def __init__(self, config):

        """
        Initialize the WikiDataset object.

        Args:
            config (dict): A dictionary containing the configuration for the dataset.
        """

        self.config = config["data"]
        self.wiki_id = config["wiki_id"]
        self.data = datasets.load_dataset(
            "wikimedia/wikipedia",
            f"20231101.{self.wiki_id}"
        )
        with resources.open_text("wqe.data.resources", "wiki_mappings.json") as f:
            self.wiki_mappings = json.load(f)[self.wiki_id]
        self.regex = None
        self.lang_id_model = None

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

    def _get_export_id(self):

        """
        Get the id string for exporting the dataset.

        Returns:
            str: The export ID.
        """

        ids = []
        if self.config["pre_filter"]["do_pre_filter"]:
            ids.append("pre_filtered")
        if self.config["partition"]["do_partition"]:
            ids.append(self.config["partition"]["partition_type"])

        return ".".join(ids)

    def _make_regex(self):

        """
        Make the regex for filtering the dataset.
        """

        script_to_add = "".join([f"\\p{{{script}}}" for script in self.wiki_mappings['scripts']])
        script_regex = fr"[\p{{P}}\p{{S}}\d{script_to_add}]*[\d{script_to_add}]+[\p{{P}}\p{{S}}\d{script_to_add}]*\s"
        self.regex = script_regex

    def _pre_filter_article(self, article):

        """
        Pre-filter an article based on the configuration.

        Args:
            article (dict): The article to filter.

        Returns:
            dict: The filtered article.
        """

        filtered = "".join(re.findall(self.regex, article['text']))
        if self.lang_id_model is not None:
            article['text'] = "".join(
                [line for line in filtered.splitlines() if len(line) > self.config["pre_filter"]["char_cutoff"] and
                 self.lang_id_model.predict(line)[0][0].split("_")[-2] == self.wiki_mappings['alpha3']]
            )
        else:
            article['text'] = "".join(
                [line for line in filtered.split("\n") if len(line) > self.config["char_cutoff"]]
            )

        return article

    def pre_filter(self):

        """
        Pre-filter the dataset.

        Returns:
            datasets.Dataset: The pre-filtered dataset.
        """

        if self.config["pre_filter"]["script_regex"]:
            self._make_regex()
        if self.config["pre_filter"]["lang_id"]:
            self.lang_id_model = fasttext.load_model(
                hf_hub_download(
                    repo_id="cis-lmu/glotlid",
                    filename="model.bin",
                    cache_dir=None
                )
            )

        self.data = self.data.map(
            lambda article: self._pre_filter_article(article)
        )

    def save(self):

        """
        Save the dataset to disk.
        """

        export_id = self._get_export_id()
        path = self.config["export"]["path"]

        if self.config["export_type"] == "hub":
            self.data.push_to_hub(
                f"{path}/{export_id}/{self.wiki_id}",
                private=True
            )
        elif self.config["export_type"] == "local":
            if not os.path.exists(f"{path}/{export_id}"):
                os.makedirs(f"{path}/{export_id}")
            self.data.save_to_disk(
                f"{path}/{export_id}/{self.wiki_id}"
            )
        else:
            print("Invalid export type. Please specify either 'hub' or 'local'.")
            print("Skipping export.")
