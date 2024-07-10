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
from datasketch import MinHash, MinHashLSH
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import insecure_hashlib
from transformers import PreTrainedTokenizerFast

from . import resources
from ..utils.maps import PARTITION_MAP

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

        logger.info(f"Loaded {self.n_docs} articles with {self.n_chars} characters (train). Wiki: {self.wiki.id}")

        return self

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

    def pre_filter(
            self,
            script_regex: bool = False,
            lang_id: bool = False,
            deduplicate_exact_match: bool = False,
            deduplicate_min_hash: bool = False,
            jaccard_threshold: float = 0.85,
            tokenizer: str = None,
            char_cutoff: int = 0,
            urls_to_remove: List[str] = None,
            num_proc: int = 1
    ) -> 'WikiLoader':

        """
        Pre-filters the dataset using the following functions:

        - `script_regex`: Removes lines from the dataset that do not contain
        any of the accepted scripts for the language (e.g. Cyrillic for English).

        - `lang_id`: Removes lines from the dataset that are not identified as
        belonging to the specified language. This is done via the GlotLID model.
        CAUTION: This is very slow and should be used sparingly, as it is not
        guaranteed to be accurate for lower-resourced languages.

        - `deduplicate_exact_match`: Removes duplicate articles by hashing
        the text of each article and removing exact match duplicates.

        - `deduplicate_min_hash`: Removes duplicate articles by computing
        the Jaccard similarity between article unigrams using MinHash-LSH,
        and filtering based on the specified threshold. Can be used in conjunction
        with a trained tokenizer, if provided. Otherwise, will lowercase
        and split on whitespace.

        - `char_cutoff`: Removes lines from the dataset that are below a certain
        character count. This is useful in conjunction with `script_regex`
        or `lang_id`, as it can remove documents that were shortened due to filtering.

        - `urls_to_remove`: Removes articles with specified URLs from the dataset.

        This method first makes a full pass through the dataset in order to apply the
        `script_regex` and `lang_id` filters, and compute hashes for articles.
        It then makes a second pass through the dataset to remove articles according
        to the `char_cutoff`, `deduplicate_exact_match`, and `deduplicate_min_hash` filters.

        The filters can be applied simultaneously...

        ```
        from wqe import WikiLoader
        loader = WikiLoader("ha")
        loader.pre_filter(script_regex=True, lang_id=True, deduplicate_exact_match=True, deduplicate_min_hash=True)
        ```

        ...or successively:

        ```
        from wqe import WikiLoader
        loader = WikiLoader("ha")
        loader.pre_filter(script_regex=True, lang_id=True)
        loader.pre_filter(deduplicate_exact_match=True, deduplicate_min_hash=True)
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
            Default is False.
        deduplicate_exact_match : bool
            Whether to deduplicate articles via exact match.
            Default is False.
        deduplicate_min_hash : bool
            Whether to deduplicate articles via MinHash-LSH.
            Default is False.
        jaccard_threshold : float
            The Jaccard similarity threshold for deduplication.
            Default is 0.85.
        tokenizer : str
            Path to tokenizer to use in computing jaccard similarity between documents.
            Optional for min_hash deduplication.
        char_cutoff : int
            The minimum number of characters required for a line to be kept.
            Default is 50.
        urls_to_remove : list
            The list of URLs to remove from the dataset.
            Useful for buggy articles such as https://xh.wikipedia.org/wiki/Phi.
        num_proc : int
            The number of processes to use for filtering.
            Default is 4.
        """

        raw_chars = self.n_chars
        raw_docs = self.n_docs

        if script_regex:
            self._make_regex()
            logger.info(f"Filtering documents for accepted scripts: {self.wiki.scripts}")

        if lang_id:
            if num_proc > 1:
                logger.warning("Language ID is not supported for multiprocessing. Setting num_proc=1.")
                num_proc = 1
            logger.info(f"Filtering documents for language: {self.wiki.alpha3}")
            logger.info(f"Loading GlotLID model.")
            self.lang_id_model = fasttext.load_model(
                hf_hub_download(
                    repo_id="cis-lmu/glotlid",
                    filename="model.bin",
                    cache_dir=None
                )
            )

        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer) if tokenizer else None

        self.data = self.data.map(
            function=self._pre_filter_article,
            fn_kwargs={
                "script_regex": script_regex,
                "regex_pattern": self.regex,
                "lang_id": lang_id,
                "model": self.lang_id_model,
                "alpha3": self.wiki.alpha3,
                "deduplicate_exact_match": deduplicate_exact_match,
                "deduplicate_min_hash": deduplicate_min_hash,
                "tokenizer": tokenizer,
                "urls_to_remove": urls_to_remove
            },
            num_proc=num_proc
        )

        if deduplicate_exact_match:
            logger.info(f"Filtering documents for exact match duplicates.")
            unique_hashes = set(self.data["train"].unique("hash"))
            frac = len(unique_hashes) / len(self.data["train"])
            logger.info(f"Fraction of exact duplicate articles: {1 - frac:.2%}")
        else:
            unique_hashes = None

        if deduplicate_min_hash:
            logger.info(f"Filtering documents for MinHash duplicates with LSH.")
            lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=256)
        else:
            lsh = None

        if char_cutoff > 0:
            logger.info(f"Character cutoff set to {char_cutoff}.")

        logger.info(f"Removing documents.")
        self.data = self.data.filter(
            function=self._remove_articles,
            fn_kwargs={
                "unique_hashes": unique_hashes,
                "lsh": lsh,
                "char_cutoff": char_cutoff
            },
            num_proc=num_proc
        )

        self.n_chars = len("".join(self.data["train"]["text"]))
        self.n_docs = len(self.data["train"])

        logger.info(f"Removed {raw_chars - self.n_chars} chars ({1.0 - self.n_chars / raw_chars:.4f}%).")
        logger.info(f"Removed {raw_docs - self.n_docs} documents ({1.0 - self.n_docs / raw_docs:.4f}%).")

        return self

    @staticmethod
    def _pre_filter_article(
            article: Dict[str, Any],
            script_regex: bool = True,
            regex_pattern: str = None,
            lang_id: bool = True,
            model: fasttext.FastText = None,
            alpha3: str = None,
            deduplicate_exact_match: bool = False,
            deduplicate_min_hash: bool = False,
            tokenizer: PreTrainedTokenizerFast = None,
            urls_to_remove: List[str] = None
    ) -> Dict[str, Any]:

        """
        Pre-filters article using regex, lang_id, exact match,
        and min-hash deduplication - in that order. Also accepts
        a list of URLs to remove from the dataset, if necessary.
        Primarily used for calling in the `datasets.Dataset.map` function.

        Parameters
        ----------
        article : dict
            The article to pre-filter.
        script_regex : bool
            Whether to filter the article for accepted scripts.
        lang_id : bool
            Whether to filter the article for the specified language.
        model : fasttext.FastText
            The GlotLID model for predicting the language of a line.
        alpha3 : str
            The ISO 639-3 code for the language.
        deduplicate_exact_match : bool
            Whether to deduplicate the article via exact match.
        deduplicate_min_hash : bool
            Whether to deduplicate the article via MinHash-LSH.
        tokenizer : PreTrainedTokenizerFast
            The tokenizer to use for computing jaccard similarity for min_hash.
        urls_to_remove : list
            The list of URLs to remove from the article.

        Returns
        -------
        dict
            The pre-filtered article.
        """

        if urls_to_remove:
            if article["url"] in urls_to_remove:
                article["text"] = ""

        if script_regex:
            assert regex_pattern is not None, "Regex pattern must be specified for script filtering."
            article["text"] = "".join(re.findall(regex_pattern, article['text']))
            cleanup_pattern = r'\s+[^\w\s]+\s+|(?<=\S)\s+(?=\.$)'
            article["text"] = re.sub(cleanup_pattern, lambda m: ' ' if m.group() else '', article["text"])

        if lang_id:
            assert model is not None, "Language ID model must be specified for language filtering."
            article["text"] = "".join(
                [line for line in article["text"].splitlines() if model.predict(line)[0][0].split("_")[-2] == alpha3]
            )

        if deduplicate_exact_match:
            article["hash"] =\
                insecure_hashlib.md5(
                    re.sub(re.compile(r"\s+"), "", article["text"]).encode("utf-8")
                ).hexdigest()

        if deduplicate_min_hash:
            if tokenizer is None:
                text = re.sub(r'[^\w\s]', '', article["text"].lower())
                text = text.split()
            else:
                text = tokenizer.tokenize(article["text"].lower())

            minhash = MinHash(num_perm=256)
            for word in text:
                minhash.update(word.encode('utf-8'))

            article["minhash"] = minhash.digest()

        return article

    @staticmethod
    def _remove_articles(
            # self,
            article: str,
            unique_hashes: set,
            lsh: MinHashLSH,
            char_cutoff: int
    ):
        """
        Apply deduplication and char_cutoff filters to dataset.
        """

        if unique_hashes:
            if not article["hash"] in unique_hashes:
                return False
            else:
                unique_hashes.remove(article["hash"])
        if lsh:
            minhash = MinHash(hashvalues=article["minhash"])
            if lsh.query(minhash):
                return False
            else:
                lsh.insert(article["id"], minhash)
        elif len(article["text"]) < char_cutoff:
            return False

        return True

    def _make_regex(self):

        """
        Makes regex for filtering the dataset for all accepted unicode scripts for a language.
        """

        scripts = "".join([f"\\p{{{script}}}" for script in self.wiki.scripts])
        script_regex = fr"[\p{{P}}\p{{S}}\d{scripts}]*[\d{scripts}]+[\p{{P}}\p{{S}}\d{scripts}]*\s"
        self.regex = script_regex

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
