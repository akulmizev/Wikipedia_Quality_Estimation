import json
import logging
import os
from importlib import resources

from tokenizers import Tokenizer, processors, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from .resources.param_map import PARAM_MAP

logger = logging.getLogger(__name__)
# logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

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
        self.experiment_id = config["experiment"]["id"]
        self.wiki_id = config["wiki_id"]

        self.tokenizer = None
        if "from_pretrained" in self.config:
            # TODO: Clean up messiness with loading from_pretrained with wiki_id.
            logging.info(f"Loading tokenizer from hub: {self.config['from_pretrained']}.{self.wiki_id}")
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{self.config['from_pretrained']}.{self.wiki_id}")
        elif "from_config" in self.config:
            logging.info("Building tokenizer from config.")
            self._build_tokenizer()
        else:
            raise ValueError("Pass a configuration for training or importing a tokenizer.")

    def __getattr__(self, attr):

        """
        Get an attribute from the tokenizer.

        Args:
            attr (str): The attribute to retrieve.

        Returns:
            Any: The attribute.
        """
        return getattr(self.tokenizer, attr)

    def __call__(self, *args, **kwargs):

        """
        Call the tokenizer.

        Args:
            *args: The positional arguments.
            **kwargs: The keyword arguments.

        Returns:
            Any: The result of the call.
        """
        return self.tokenizer(*args, **kwargs)

    def _build_tokenizer(self):

        try:
            train_config = self.config["from_config"]
        except KeyError("Pass a configuration for training a tokenizer via 'from_config' key."):
            exit()

        self.tokenizer = Tokenizer(PARAM_MAP["model"][train_config["model"]]())
        self.tokenizer.normalizer = PARAM_MAP["normalizer"][train_config["normalizer"]]()
        if len(train_config["pre_tokenizer"]) > 1:
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [PARAM_MAP["pre_tokenizer"][pt]() for pt in train_config["pre_tokenizer"]]
            )
        else:
            self.tokenizer.pre_tokenizer = PARAM_MAP["pre_tokenizer"][train_config["pre_tokenizer"][0]]()
        self.tokenizer.decoder = PARAM_MAP["decoder"][train_config["decoder"]]()

        if train_config["vocab_size"] == "auto":
            with resources.open_text("wqe.tokenizer.resources", "predicted_vocab.json") as f:
                predicted_vocab = json.load(f)
            train_config["vocab_size"] = predicted_vocab[self.wiki_id]

        self.tokenizer.trainer = PARAM_MAP["trainer"][train_config["trainer"]](
            vocab_size=train_config["vocab_size"],
            special_tokens=train_config["special_tokens"],
            unk_token=train_config["unk_token"]
        )

        if train_config["post_processor"]:
            self.tokenizer.post_processor = processors.TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
            )

    def train(self, dataset, batch_size=1000):
        """
        Train the tokenizer on a dataset.

        Args:
            dataset (datasets.Dataset): The dataset to train the tokenizer on.
            batch_size (int): The batch size.
        """
        logging.info(f"Training tokenizer on {len(dataset)} samples...")
        self.tokenizer.train_from_iterator(
            self.batch_iterator(dataset, batch_size=batch_size),
            trainer=self.tokenizer.trainer,
            length=len(dataset)
        )

        logging.info(f"Trained a tokenizer with vocab size: {self.tokenizer.get_vocab_size()}")

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)

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

    def convert_to_fast(self):
        """
        Convert the tokenizer to a PreTrainedTokenizerFast object.
        """

        return PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)

    def get_base_tokenizer(self):
        """
        Get the base tokenizer object.

        Returns:
            Tokenizer: The base tokenizer object.
        """

        # TODO - This is a temporary fix. Need to find a better way to handle this.

        return self.tokenizer

    def save(self):
        """
        Save the tokenizer to a file.

        """

        path = self.config["export"]["path"]
        export_type = self.config["export"]["export_type"]

        if export_type == "hub":
            # self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
            logging.info(f"Pushing tokenizer to hub: {path}/{self.experiment_id}.{self.wiki_id}")
            self.tokenizer.push_to_hub(
                f"{path}/{self.experiment_id}.{self.wiki_id}",
                use_temp_dir=True,
                repo_name=path,
                private=True
            )
        elif export_type == "local":
            logging.info(f"Saving tokenizer to: {path}/{self.experiment_id}/{self.wiki_id}")
            if not os.path.exists(f"{path}/{self.experiment_id}"):
                os.makedirs(f"{path}/{self.experiment_id}")
            self.tokenizer.save_pretrained(f"{path}/{self.experiment_id}/{self.wiki_id}")
        else:
            raise ValueError("Invalid export type.")


class FastTokenizerFromConfig(PreTrainedTokenizerFast):

    @classmethod
    def train_from_config(cls, dataset, config_path, batch_size=1000):

        config = json.load(open(config_path))

        tokenizer = Tokenizer(
            PARAM_MAP["model"][config["model"]]()
        )

        tokenizer.add_special_tokens(list(config["special_tokens"].values()))
        tokenizer.normalizer = PARAM_MAP["normalizer"][config["normalizer"]]()
        if isinstance(config["pre_tokenizer"], list):
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [PARAM_MAP["pre_tokenizer"][pt]() for pt in config["pre_tokenizer"]]
            )
        else:
            tokenizer.pre_tokenizer = PARAM_MAP["pre_tokenizer"][config["pre_tokenizer"][0]]()
        tokenizer.decoder = PARAM_MAP["decoder"][config["decoder"]]()

        # if config["vocab_size"] == "auto":
        #     with resources.open_text("wqe.tokenizer.resources", "predicted_vocab.json") as f:
        #         predicted_vocab = json.load(f)
        #     config["vocab_size"] = predicted_vocab[self.wiki_id]

        trainer = PARAM_MAP["trainer"][config["trainer"]](
            vocab_size=config["vocab_size"],
            special_tokens=list(config["special_tokens"].values()),
            unk_token=config["unk_token"]
        )

        if config["post_processor"]:
            tokenizer.post_processor = processors.TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
            )

        tokenizer.train_from_iterator(
            cls.batch_iterator(dataset, batch_size=batch_size),
            trainer=trainer,
            length=len(dataset)
        )

        return cls(tokenizer_object=tokenizer, **config["special_tokens"])

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
