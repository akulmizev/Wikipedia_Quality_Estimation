import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Union

from huggingface_hub import HfApi

from ..utils.config import MainConfig
from ..data.loader import WikiLoader
from ..tokenizer.tokenizer import PreTrainedTokenizerFast
from ..model.pretrain import MLM
from ..model.finetune import Tagger, Classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:

	def __init__(
			self,
			config: Union[Dict[str, Any], MainConfig]
	):

		"""
		Initialize the ExperimentRunner object.

		Args:
			config (dict): A dictionary containing the configuration for the experiment_cfg.
		"""

		self.experiment_cfg = None
		self.dataset_cfg = None
		self.tokenizer_cfg = None
		self.pretrain_cfg = None
		self.finetune_cfg = None

		self.__dict__.update(**config.__dict__)

		if not self.experiment_cfg:
			raise ValueError("Experiment configuration is required.")

		self.wiki_id = self.experiment_cfg.wiki_id
		self.experiment_path = None

		if self.experiment_cfg.local_path:
			self.experiment_path = f"{self.experiment_cfg.local_path}/{self.experiment_cfg.experiment_id}"
			os.makedirs(self.experiment_path, exist_ok=True)

		if self.experiment_cfg.hub_path:
			self.api = HfApi()

	def process_dataset(self, cfg):

		dataset = WikiLoader(self.wiki_id)
		dataset.load_dataset() if cfg.load_path is None else dataset.load_dataset(cfg.load_path)

		if cfg.pre_filter:
			dataset.pre_filter(**asdict(cfg.pre_filter))

		if cfg.partition:
			dataset.apply_partition(**asdict(cfg.partition))

		if cfg.split:
			dataset.generate_splits(**asdict(cfg.split))

		if cfg.export:
			dataset.save(f"{self.experiment_path}/data/{self.experiment.wiki_id}")
			if cfg.push_to_hub:
				dataset.push_to_hub(self.experiment_cfg.hub_path, self.wiki_id)

		return dataset

	def process_tokenizer(self, cfg, dataset=None):

		tokenizer = None

		if cfg.load_path:
			logger.info(f"Loading tokenizer from {cfg.load_path}")
			tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.load_path)

		elif cfg.parameters:
			assert dataset is not None, "Dataset is required for training tokenizer."
			tokenizer = PreTrainedTokenizerFast.train_from_config(
				dataset["train"]["text"],
				**asdict(cfg.parameters)
			)
			if cfg.export:
				tokenizer.save_pretrained(f"{self.experiment_path}/tokenizer")
				if cfg.push_to_hub:
					tokenizer.push_to_hub(self.experiment_cfg.hub_path)
		else:
			raise ValueError("Tokenizer configuration is required.")

		return tokenizer

	def run_experiment(self):

		if self.dataset_cfg:
			dataset = self.process_dataset(self.dataset_cfg)

		if self.tokenizer_cfg:
			tokenizer = self.process_tokenizer(self.tokenizer_cfg, dataset)

		pass

