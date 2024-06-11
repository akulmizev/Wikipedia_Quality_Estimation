import logging
from typing import Any, Dict, Optional, Union

import torch
import wandb

from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInitMixin:

    """
    Mixin class for initializing model parameters and configurations.
    Useful to define common parameters here and not in the base model class.

    Parameters
    ----------
    model_type : str
        Type of the model (e.g., 'bert', 'roberta', etc.).
    task : str
        Task type ('mlm', 'clm', 'pos', 'ner', or 'classification').
    num_train_epochs : int
        Number of training epochs.
    max_length : int, optional
        Maximum length of input sequences (default is 512).
    batch_size : int, optional
        Batch size for training and evaluation (default is 8).
    lr : float, optional
        Learning rate for the optimizer (default is 1e-3).
    padding_strategy : str, optional
        Padding strategy for input sequences ('max_length' or 'batch') (default is 'max_length').
    mask_prob : float, optional
        Probability for masking tokens during masked language modeling (default is 0.15).
    num_eval_steps : int, optional
        Number of steps between evaluation during training (default is None).
        If None, evaluation is performed at the end of each epoch.
    checkpoint_path : Union[str, None], optional
        Path to save model checkpoints during training (default is None).

    Attributes
    ----------
    *see Parameters*
    accelerator : Accelerator
        Accelerator instance for distributed training and optimization.
    label_set : set or None
        Set of labels for the task (None if not applicable).
    wandb : bool
        Flag indicating whether to use Weights & Biases for logging.
    """

    MAX_GPU_BATCH_SIZE = 128 #128 is good batch_size 1024, seqlen 128, and lr 1e-3

    def __init__(
            self,
            model_type: str,
            task: str,
            num_train_epochs: int,
            max_length: Optional[int] = 128,
            batch_size: Optional[int] = 8,
            lr: Optional[float] = 1e-3,
            padding_strategy: Optional[str] = "longest",
            mask_prob: Optional[float] = 0.15,
            grad_accumulation_steps: Optional[int] = 1,
            mixed_precision: Optional[str] = "no",
            num_eval_steps: Optional[int] = None,
            checkpoint_path: Optional[Union[str, None]] = None,
    ):
        self.model_type = model_type
        self.task = task
        self.num_train_epochs = num_train_epochs
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.padding_strategy = padding_strategy
        self.mask_prob = mask_prob
        self.grad_accumulation_steps = grad_accumulation_steps
        self.mixed_precision = mixed_precision
        self.num_eval_steps = num_eval_steps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.checkpoint_path = checkpoint_path
        self.label_set = None
        self.wandb = False

        self._check_params()

        self.accelerator = Accelerator(
            project_dir=self.checkpoint_path if self.checkpoint_path else None,
            mixed_precision=self.mixed_precision,
            device_placement=True
        )

        if self.batch_size > self.MAX_GPU_BATCH_SIZE:
            self.grad_accumulation_steps = self.batch_size // self.MAX_GPU_BATCH_SIZE
            self.batch_size = self.MAX_GPU_BATCH_SIZE

        if self.accelerator.mixed_precision == "fp8":
            self.pad_to_multiple_of = 16
        elif self.accelerator.mixed_precision != "no":
            self.pad_to_multiple_of = 8
        else:
            self.pad_to_multiple_of = None

    def init_wandb(
            self,
            project: str,
            entity: str,
            config: Dict[str, Any]
    ):

        """
        Initialize Weights & Biases for logging.

        Parameters
        ----------
        project : str
            Project name for wandb.
        entity : str
            Entity name for wandb.
        config : Dict[str, Any]
            Training parameters (usually same as used for __init__).
        """

        wandb.init(
            project=project,
            entity=entity,
            config=config,
            dir=None
        )

        self.wandb = True

    def _check_params(self):

        """
        Check the validity of provided parameters.
        Probably needs more work.
        """

        assert self.task in ["mlm", "clm", "pos", "ner", "classification"], \
            (f"Provided invalid task type: {self.task}."
             f"Must be one of 'mlm', 'clm', 'pos', 'ner', 'classification'.")

        assert self.padding_strategy in ["max_length", "longest"], \
            (f"Provided invalid padding type: {self.padding_strategy}. "
             f"Must be one of 'max_length', 'longest'.")
