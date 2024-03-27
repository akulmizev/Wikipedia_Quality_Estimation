import json
from importlib import resources
from tokenizers import Tokenizer, processors, pre_tokenizers

from .resources.param_map import PARAM_MAP


class WikiTokenizerFromConfig:
    """
    Base class for training a tokenizer on a Wikipedia dump.
    """

    def __init__(self, config):
        """
        Initialize the WikiTokenizer object.

        Args:
            config (dict): A dictionary containing the configuration for the tokenizer.
        """

        self.config = config["tokenizer"]
        self.wiki_id = config["wiki_id"]
        # super().__init__(PARAM_MAP["model"][self.config["model"]]())
        self.tokenizer = Tokenizer(PARAM_MAP["model"][self.config["model"]]())
        self.tokenizer.normalizer = PARAM_MAP["normalizer"][self.config["normalizer"]]()
        if len(self.config["pre_tokenizer"]) > 1:
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [PARAM_MAP["pre_tokenizer"][pt]() for pt in self.config["pre_tokenizer"]]
            )
        else:
            self.tokenizer.pre_tokenizer = PARAM_MAP["pre_tokenizer"][self.config["pre_tokenizer"][0]]()
        self.tokenizer.decoder = PARAM_MAP["decoder"][self.config["decoder"]]()

        if self.config["vocab_size"] == "auto":
            with resources.open_text("wqe.tokenizer.resources", "predicted_vocab.json") as f:
                predicted_vocab = json.load(f)
            self.config["vocab_size"] = predicted_vocab[self.wiki_id]

        self.tokenizer.trainer = PARAM_MAP["trainer"][self.config["trainer"]](
            vocab_size=self.config["vocab_size"],
            special_tokens=self.config["special_tokens"],
            unk_token=self.config["unk_token"]
        )

        if self.config["post_processor"]:
            self.tokenizer.post_processor = processors.TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
            )

    def __getattr__(self, attr):

        """
        Get an attribute from the tokenizer.

        Args:
            attr (str): The attribute to retrieve.

        Returns:
            Any: The attribute.
        """
        return getattr(self.tokenizer, attr)

    def train(self, dataset, batch_size=1000):
        """
        Train the tokenizer on a dataset.

        Args:
            dataset (datasets.Dataset): The dataset to train the tokenizer on.
            batch_size (int): The batch size.
        """
        self.tokenizer.train_from_iterator(
            self.batch_iterator(dataset, batch_size=batch_size),
            trainer=self.tokenizer.trainer,
            length=len(dataset)
        )

    @staticmethod
    def batch_iterator(dataset, batch_size=1000):
        """
        Iterate over the dataset in batches.

        Args:
            dataset (datasets.Dataset): The dataset to iterate over.
            batch_size (int): The batch size.

        Returns:
            list: A list of batches of the dataset.
        """
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]["text"]
