from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional


@dataclass
class Load:
    method: str
    path: Optional[str] = None

    def __post_init__(self):
        if self.method not in ["local", "hub", "raw", "config", "pretrained"]:
            raise ValueError(f"Invalid load method: {self.method}")
        if self.method in ["local", "hub", "pretrained"] and not self.path:
            raise ValueError(f"Path is required for {self.method} method")


@dataclass
class PreFilter:
    script_regex: Optional[bool] = None
    lang_id: Optional[bool] = None
    char_cutoff: Optional[int] = None


@dataclass
class Partition:
    method: str
    metric_type: str
    quality: bool
    tokenizer: Optional[str] = None
    all_partitions_join_method: Optional[str] = None

    def __post_init__(self):
        if self.method not in ["mean_cutoff", "balanced_docs", "balanced_chars"]:
            raise ValueError(f"Invalid partition method: {self.method}")
        if self.metric_type == "all" and not self.all_partitions_join_method:
            raise ValueError(f"Join method is required for 'all' partition type")


@dataclass
class Split:
    train: float
    test: float
    seed: Optional[int] = 12345
    shuffle: Optional[bool] = True


@dataclass
class TokenizerParameters:
    model: str = "unigram"
    normalizer: str = "nkfc"
    pre_tokenizer: Union[str, List[str]] = "byte_level"
    decoder: str = "byte_level"
    vocab_size: Union[int, str] = "auto"
    trainer: str = "unigram"
    special_tokens: Dict[str, str] = field(
        default_factory=lambda: {
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]"
        }
    )
    unk_token: str = "[UNK]"
    post_processor: Optional[bool] = False
    min_frequency: Optional[int] = None

    # TODO: Add type-checking for parameters


@dataclass
class TrainingParameters:
    model_type: str
    task: str
    num_train_epochs: int
    max_length: int = 512
    batch_size: int = 8
    lr: float = 1e-5
    mask_prob: Optional[float] = None
    eval_steps: Optional[int] = None

    def __post_init__(self):
        if self.task not in ["mlm", "ner"]:
            raise ValueError(f"Invalid task: {self.task}")

    # TODO: Consolidate for fine-tuning, which might have different params


@dataclass
class Experiment:
    wiki_id: str
    experiment_id: str
    wandb_entity: str
    local_path: str
    hub_path: Optional[str]


@dataclass
class Dataset:
    load: Dict[str, str]
    export: bool
    push_to_hub: bool = False
    pre_filter: Optional[Dict[str, str]] = None
    partition: Optional[Dict[str, str]] = None
    split: Optional[Dict[str, str]] = None

    def __post_init__(self):
        self.load = Load(**self.load)
        if self.pre_filter:
            self.pre_filter = PreFilter(**self.pre_filter)
        if self.partition:
            self.partition = Partition(**self.partition)
        if self.split:
            self.split = Split(**self.split)


@dataclass
class Tokenizer:
    load: Dict[str, str]
    export: bool
    push_to_hub: bool = False
    parameters: Optional[Dict[str, Union[str, int, bool, Dict[str, str]]]] = None

    def __post_init__(self):
        self.load = Load(**self.load)
        if self.load.method == "config" and not self.parameters:
            raise ValueError("Parameters are required for config-based tokenizer training.")
        if self.parameters:
            self.parameters = TokenizerParameters(**self.parameters)


@dataclass
class Pretrain:
    load: Dict[str, str]
    export: bool
    push_to_hub: bool = False
    do_train: bool = False
    training_parameters: Optional[Dict[str, Union[str, int, float, bool]]] = None
    test_data: Optional[Dict[str, str]] = None

    def __post_init__(self):
        self.load = Load(**self.load)
        if self.training_parameters:
            self.training_parameters = TrainingParameters(**self.training_parameters)
        if self.test_data:
            self.test_data = Load(**self.test_data)

@dataclass
class Finetune:
    load: Dict[str, str]
    export: bool
    push_to_hub: bool = False
    do_train: bool = False
    training_parameters: Optional[Dict[str, Union[str, int, float, bool]]] = None

    def __post_init__(self):
        self.load = Load(**self.load)
        if self.training_parameters:
            self.training_parameters = TrainingParameters(**self.training_parameters)

def parse_config(config):
    experiment = Experiment(**config["experiment"])
    data = Dataset(**config["data"]) if "data" in config else None
    tokenizer = Tokenizer(**config["tokenizer"]) if "tokenizer" in config else None
    pretrain = Pretrain(**config["pretrain"]) if "pretrain" in config else None
    finetune = Finetune(**config["finetune"]) if "finetune" in config else None

    return experiment, data, tokenizer, pretrain, finetune
