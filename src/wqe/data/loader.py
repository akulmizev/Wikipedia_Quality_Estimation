import json
import logging
import regex as re
import os
import importlib.resources as pkg_resources

from dataclasses import dataclass
from typing import List, Dict, Any, Union

import datasets
import fasttext
from datasets import load_dataset, DatasetDict
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import hf_hub_download

from . import resources
from ..utils.maps import PARTITION_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

datasets.disable_caching()
# datasets.utils.logging.disable_progress_bar()


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
    regex : str
        The regex for filtering the dataset for accepted scripts.
    lang_id_model : fasttext.FastText._FastText
        The GlotLID model for predicting the language of a line.
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
        self.regex = None
        self.lang_id_model = None
        self.n_chars = 0
        self.n_docs = 0

    def __getattr__(self, attr: str):
        return getattr(self.data, attr)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __repr__(self):
        return f"WikiLoader(wiki_id='{self.wiki.id}', n_docs={self.n_docs}, n_chars={self.n_chars})"

    def __str__(self):
        return f"WikiLoader for {self.wiki.language} ({self.wiki.alpha3})\n" \
               f"Wiki ID: {self.wiki.id}\n" \
               f"Articles: {self.n_docs}\n" \
               f"Characters: {self.n_chars}\n" \
               f"Scripts: {', '.join(self.wiki.scripts)}"

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

    def load_dataset(
            self,
            load_path=None,
            split=None
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
        """

        if load_path:
            try:
                self.data = load_dataset(load_path, self.wiki.id)
            except (DatasetNotFoundError, FileNotFoundError):
                raise DatasetNotFoundError(f"Could not find dataset at {load_path}/{self.wiki.id}.")
        else:
            self.data = load_dataset(
                "wikimedia/wikipedia",
                f"20231101.{self.wiki.id}",
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

        logger.info(f"Loaded {self.n_docs} articles with {self.n_chars} characters (train). Wiki: {self.wiki.id}")

        return self

    @classmethod
    def from_dataset_dict(
            cls,
            wiki_id: str,
            dataset_dict: DatasetDict
    ) -> 'WikiLoader':

        """
        Initializes a WikiLoader instance from a datasets.DatasetDict.

        Parameters
        ----------
        wiki_id : str
            The language id used by Wikipedia. For example, "as" for Assamese.
            Can be found in the `Wiki` column here:
            https://meta.wikimedia.org/wiki/List_of_Wikipedias.
        dataset_dict : DatasetDict
            The dataset dictionary to load the dataset from.

        Returns
        -------
        WikiLoader
            The initialized WikiLoader instance.
        """

        instance = cls(wiki_id)  # Create an instance by calling __init__
        instance.data = dataset_dict
        instance.regex = None
        instance.lang_id_model = None

        instance.n_chars = len("".join(instance.data["train"]["text"]))
        instance.n_docs = len(instance.data["train"])

        logger.info(f"Loaded {instance.n_docs} articles with {instance.n_chars} characters (train).")

        return instance

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

        logger.info("Generating dataset splits...")

        self.data = self.data['train'].train_test_split(
            test_size=test_size,
            shuffle=shuffle,
            seed=seed
        )

        self.n_chars = len("".join(self.data["train"]["text"]))
        self.n_docs = len(self.data["train"])

        logger.info(f"Generated new train split with {self.n_docs} articles and {self.n_chars} characters.")

        return self

    def _make_regex(self):

        """
        Makes regex for filtering the dataset for all accepted unicode scripts for a language.
        """

        scripts = "".join([f"\\p{{{script}}}" for script in self.wiki.scripts])
        script_regex = fr"[\p{{P}}\p{{S}}\d{scripts}]*[\d{scripts}]+[\p{{P}}\p{{S}}\d{scripts}]*\s"
        self.regex = script_regex

    def pre_filter(
            self,
            script_regex: bool = False,
            lang_id: bool = False,
            char_cutoff: int = 50
    ) -> 'WikiLoader':

        """
        Pre-filters the dataset using both `regex` and `lang_id` functions,
        then removes documents with less characters than `char_cutoff`.

        Lang-id uses GlotLID (https://github.com/cisnlp/GlotLID)
        to predict the language of each line in the dataset.
        There is no gpu acceleration available for this model
        (to my knowledge), so running it can be very slow.

        TODO: Make lang_id filtering faster (somehow).

        Parameters
        ----------
        script_regex : bool
            Whether to filter the dataset for accepted scripts.
            Default is False.
        lang_id : bool
            Whether to filter the dataset for the specified language.
            Default is False.
        char_cutoff : int
            The minimum number of characters required for a line to be kept.
            Default is 50.
        """

        raw_chars = self.n_chars
        raw_docs = self.n_docs

        if script_regex:
            self._make_regex()
            logger.info(f"Filtering documents for accepted scripts: {self.wiki.scripts}")

        if lang_id:
            logger.info(f"Filtering documents for language: {self.wiki.alpha3}")
            logger.info(f"Loading GlotLID model.")
            self.lang_id_model = fasttext.load_model(
                hf_hub_download(
                    repo_id="cis-lmu/glotlid",
                    filename="model.bin",
                    cache_dir=None
                )
            )

        self.data = self.data.map(
            lambda example: self._pre_filter_article(example, script_regex, lang_id)
        )

        logger.info(f"Removing documents shorter than {char_cutoff} characters.")
        self.data = self.data.filter(
            lambda example: len(example['text']) >= char_cutoff
        )

        self.n_chars = len("".join(self.data["train"]["text"]))
        self.n_docs = len(self.data["train"])

        logger.info(f"Removed {raw_chars - self.n_chars} chars ({1.0 - self.n_chars / raw_chars:.4f}%).")
        logger.info(f"Removed {raw_docs - self.n_docs} documents shorter than {char_cutoff} characters.")

        return self

    def _pre_filter_article(
            self,
            article: Dict[str, Any],
            script_regex: bool = True,
            lang_id: bool = True,
    ) -> Dict[str, Any]:

        """
        Pre-filters article using regex, lang_id, and char_cutoff.
        Primarily used for calling in the `datasets.Dataset.map` function.

        Parameters
        ----------
        article : dict
            The article to pre-filter.
        script_regex : bool
            Whether to filter the article for accepted scripts.
        lang_id : bool
            Whether to filter the article for the specified language.

        Returns
        -------
        dict
            The pre-filtered article.
        """

        if script_regex:
            article["text"] = "".join(re.findall(self.regex, article['text']))

        if lang_id:
            article['text'] = "".join(
                [line for line in article['text'].splitlines() if self._predict_lang_id(line) == self.wiki.alpha3]
            )

        return article

    def _predict_lang_id(self, line: str) -> str:
        """
        Predicts the language of a line using GlotLID.
        Just a helper function for `_pre_filter_article`.

        Parameters
        ----------
        line : str
            The string to predict the language of.

        Returns
        -------
        str
            The predicted alpha3 code.
        """

        if self.lang_id_model is None:
            raise ValueError("Language ID model not loaded. Please set `lang_id=True` in `pre_filter`.")

        return self.lang_id_model.predict(line)[0][0].split("_")[-2]

    def apply_partition(
            self,
            metric: Union[List[str], str],
            method: str = "balanced_chars",
            quality: bool = True,
            join_method: str = "intersection",
            tokenizer: Union[str, None] = None
    ):

        """
        Updates the dataset with a partition chosen by the specified metric.

        If multiple metrics are specified, the intersection or union
        of the returned document indices is used.

        TODO: eventually need to move much of this to the `wqe.dataset.partition` module.

        Parameters
        ----------
        method : str
            The method for choosing the boundary at which to split high-quality
            and low-quality partitions. Default is 'balanced_chars', which allocates
            approximately half of the dataset's total characters to each partition.
            Also supported:
            - 'mean_cutoff': split based on the mean value of the metric
            - 'median_cutoff': split based on the median value of the metric
            - 'balanced_docs': allocates equal number of documents to each partition
        metric : list of str or str
            The metric(s) to use for partitioning the dataset.
        quality : bool
            Whether to return the higher-quality partition or the lower-quality partition.
            Default is True for higher-quality.
        join_method : str
            If a list of metrics is specified, specifies how to join them.
            Set operations are performed on the dataset indices returned by each metric.
            Choice between 'intersection' and 'union'.
        tokenizer : str
            The tokenizer to use for partitioning the dataset. Required for certain metrics.
        """

        raw_chars = self.n_chars
        raw_docs = self.n_docs

        if isinstance(metric, str):
            logger.info(f"Partitioning dataset by {metric}...")
            partition = PARTITION_MAP[metric](
                method=method,
                quality=quality,
                tokenizer=tokenizer
            )

            partition_indices = partition(self.data["train"])
            self.data["train"] = self.data["train"].select(partition_indices)

        else:
            logger.info(f"Partitioning dataset by {', '.join(metric)}...")

            partition_indices = []
            for met in metric:
                partition = PARTITION_MAP[met](
                    method=method,
                    quality=quality,
                    tokenizer=tokenizer
                )
                partition_indices.append(partition(self.data["train"]))

            if join_method == "intersection":
                partition_indices = list(set.intersection(*map(set, partition_indices)))
            elif join_method == "union":
                partition_indices = list(set.union(*map(set, partition_indices)))
            else:
                raise ValueError("Invalid join method. Please specify either 'intersection' or 'union'.")

            self.data["train"] = self.data["train"].select(partition_indices)

        self.n_chars = len("".join(self.data["train"]["text"]))
        self.n_docs = len(self.data["train"])

        logger.info(f"Removed {raw_chars - self.n_chars} chars ({1.0 - self.n_chars / raw_chars:.4f}%).")
        logger.info(f"Removed {raw_docs - self.n_docs} docs ({1.0 - self.n_docs / raw_docs:.4f}%).")

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
