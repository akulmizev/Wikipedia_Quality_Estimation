import logging

import torch
import wandb

from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelFromConfig:

    def __init__(self,
                 config,
                 export_path=None,
                 **kwargs):

        self.__dict__.update(config.__dict__)
        # self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.torch_dtype = torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.export_path = export_path
        self.accelerator = Accelerator(project_dir="self.export_path") if self.export_path else Accelerator()
        self.model = None
        self.collator = None
        self.wandb = False

    def init_wandb(self, project, entity, parameters):
        wandb.init(
            project=project,
            entity=entity,
            config={**parameters.__dict__}
        )

        self.wandb = True

    def _tokenize_and_collate(self, dataset):
        raise NotImplementedError("Subclasses must implement this method.")

    def _eval_loop(self, loader):
        raise NotImplementedError("Subclasses must implement this method.")

    def train(self, dataset):
        raise NotImplementedError("Subclasses must implement this method.")

    def test(self, dataset):
        raise NotImplementedError("Subclasses must implement this method.")