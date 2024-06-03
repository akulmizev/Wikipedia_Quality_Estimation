import logging
from dataclasses import asdict
from typing import Optional

import torch
import wandb

from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInitMixin:
    def __init__(
            self,
            export_path: str,
            model_type: str,
            num_train_epochs: int,
            max_length: int = 512,
            batch_size: int = 8,
            lr: float = 1e-3,
            save_checkpoint: bool = False,
            mask_prob: Optional[float] = None,
            eval_steps: Optional[int] = None
    ):
        self.model_type = model_type
        self.num_train_epochs = num_train_epochs
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.save_checkpoint = save_checkpoint
        self.mask_prob = mask_prob
        self.eval_steps = eval_steps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.export_path = export_path
        self.accelerator = Accelerator(project_dir=self.export_path) if self.export_path else Accelerator()
        self.wandb = False

    def init_wandb(self, project, entity, config):
        wandb.init(
            project=project,
            entity=entity,
            config=asdict(config)
        )

        self.wandb = True
