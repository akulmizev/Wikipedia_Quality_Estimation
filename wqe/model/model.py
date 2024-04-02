import torch

from accelerate import Accelerator
from datasets import load_dataset
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from transformers import DebertaConfig
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW

class WikiModelFromConfig:
    def __init__(self, config):
        self.config = config
        self.model = None

    def get_model(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def _create_model(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def train(self):
        raise NotImplementedError("Subclasses must implement this method.")

class WikiMLM(WikiModelFromConfig):
    def __init__(self, config):
        super().__init__(config)
        self.model = self._create_model()

    def get_model(self):
        return self.model

    def _create_model(self):

        if self.config["model"]["from_config"]:
            model = AutoModelForMaskedLM.from_config(self.config["model"]["config"])

        raise NotImplementedError("Subclasses must implement this method.")

    def train(self):
        raise NotImplementedError("Subclasses must implement this method.")