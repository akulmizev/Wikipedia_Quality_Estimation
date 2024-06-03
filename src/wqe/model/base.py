import logging

from abc import ABC, abstractmethod
from dataclasses import asdict

import evaluate
import tqdm
import torch
import wandb

from accelerate import Accelerator
from transformers import CONFIG_MAPPING
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW

from .mixins import ModelInitMixin
from ..utils.maps import TASK_TO_MODEL_AND_COLLATOR_MAPPING as TASK_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# class ModelFromConfig:
#
#     def __init__(self,
#                  config,
#                  export_path=None,
#                  **kwargs):
#         self.__dict__.update(config.__dict__)
#         # self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
#         self.torch_dtype = torch.float32
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.export_path = export_path
#         self.accelerator = Accelerator(project_dir=self.export_path) if self.export_path else Accelerator()
#         self.model = None
#         self.collator = None
#         self.wandb = False
#
#     def init_wandb(self, project, entity, parameters):
#         wandb.init(
#             project=project,
#             entity=entity,
#             config={**parameters.__dict__}
#         )
#
#         self.wandb = True
#
#     def _tokenize_and_collate(self, dataset):
#         raise NotImplementedError("Subclasses must implement this method.")
#
#     def _eval_loop(self, loader):
#         raise NotImplementedError("Subclasses must implement this method.")
#
#     def train(self, dataset):
#         raise NotImplementedError("Subclasses must implement this method.")
#
#     def test(self, dataset):
#         raise NotImplementedError("Subclasses must implement this method.")


class ModelFromConfig(ModelInitMixin):
    def __init__(
            self,
            config,
            export_path=None
    ):
        super().__init__(export_path=export_path, **asdict(config))
        self.model = None
        self.collator = None
        # self.metrics = self._init_metrics()

    def _tokenize_and_collate(self, dataset):
        raise NotImplementedError("Subclasses must implement this method.")

    def _init_metrics(self):
        pass

    def _prepare_for_training(self, dataset):

        splits = dataset.keys()
        if "train" not in splits:
            raise ValueError("Train split must be present in the dataset_cfg.")

        logger.info("Tokenizing and batching datasets...")
        loaders = {split: self._tokenize_and_collate(dataset[split]) for split in splits}
        loaders["train"].shuffle = True

        self.num_train_steps = self.num_train_epochs * len(loaders["train"])
        self.eval_steps = len(loaders["train"]) if self.eval_steps is None else self.eval_steps

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.lr),
            weight_decay=0.05
        )

        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=self.num_train_steps
        )

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        for k, loader in loaders.items():
            loaders[k] = self.accelerator.prepare(loader)

        self.accelerator.register_for_checkpointing(self.scheduler)
        self.model.gradient_checkpointing_enable()

        return loaders

    def _eval_loop(self, loader):
        pass

    def train(self, dataset, eval_split="validation"):

        loaders = self._prepare_for_training(dataset)
        progress_bar = tqdm(range(self.num_train_steps))
        running_loss = torch.inf

        for epoch in range(self.num_train_epochs):
            self.model.train()
            for i, batch in enumerate(loaders["train"]):
                outputs = self.model(**batch)
                loss = outputs.loss
                if wandb:
                    wandb.log({"loss": loss.item()})
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                if i+1 % self.eval_steps == 0:
                    if eval_split in loaders:
                        scores = self._eval_loop(loaders[eval_split])
                        for score in scores:
                            # logger.info(f"{k} at epoch {epoch}: {v}")
                            if self.wandb:
                                wandb.log(score)
                        self.model.train()

            if self.save_checkpoint:
                loss = 0.0
                for i, batch in enumerate(loaders[eval_split]):
                    outputs = self.model(**batch)
                    loss += outputs.loss.item()
                eval_loss = loss / len(loaders[eval_split])
                if eval_loss < running_loss:
                    logger.info(f"Saving model checkpoint at epoch {epoch}.")
                    self.model.save_pretrained(self.export_path)
                    running_loss = eval_loss

    def test(self, dataset):

        loader = self._tokenize_and_collate(dataset)
        loader = self.accelerator.prepare(loader)
        scores = self._eval_loop(loader)
        for score in scores:
            if self.wandb:
                wandb.log(score)