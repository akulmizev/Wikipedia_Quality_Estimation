import os

from huggingface_hub import HfApi

from ..data.data import WikiLoader
from ..tokenizer.tokenizer import TokenizerFromConfig
from ..model.pretrain import MLM
from ..model.finetune import Tagger, Classifier


class ExperimentRunner:

    def __init__(self, config, **kwargs):

        """
        Initialize the ExperimentRunner object.

        Args:
            config (dict): A dictionary containing the configuration for the experiment.
        """

        self.experiment = None
        self.data = None
        self.tokenizer = None
        self.pretrain = None
        self.finetune = None

        self.__dict__.update(**config.__dict__)

        if not self.experiment:
            raise ValueError("Experiment configuration is required.")

        self.wiki_id = self.experiment.wiki_id
        self.experiment_path = None

        if self.experiment.local_path:
            self.experiment_path = f"{self.experiment.local_path}/{self.experiment.experiment_id}"
            os.makedirs(self.experiment_path, exist_ok=True)

        if self.experiment.hub_path:
            self.api = HfApi()

    def process_dataset(self, data_cfg):

        dataset = WikiLoader(self.wiki_id, )
        if data_cfg.pre_filter:
            dataset.do_pre_filter()
        if data_cfg.partition:
            dataset.apply_partition()
        if data_cfg.split:
            dataset.generate_splits()
        if data_cfg.export:
            dataset.save(f"{self.experiment_path}/data/{self.experiment.wiki_id}")
            # if data_cfg.push_to_hub:
            #     data = load_dataset(f"{self.experiment_path}/data/{self.experiment.wiki_id}")
            #     data.push_to_hub(f"{self.experiment.hub_path}/{self.experiment.experiment_id}",
            #                      config_name=f"{self.experiment.wiki_id}",
            #                      private=True)

        return dataset

    def run_experiment(self):

        if self.data:
            dataset = self.process_dataset(self.data)
