from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, Optional


@dataclass
class PreFilter:
    script_regex: Optional[bool] = False
    lang_id: Optional[bool] = False
    char_cutoff: Optional[int] = 0
    deduplicate_exact_match: Optional[bool] = False
    deduplicate_min_hash: Optional[bool] = False
    jaccard_threshold: Optional[float] = 0.85
    tokenizer: Optional[str] = None
    urls_to_remove: Optional[List[str]] = None
    num_proc: Optional[int] = 1


@dataclass
class Partition:
    metric: Union[str, List[str]]
    method: Optional[str] = "balanced_chars"
    quality: Optional[bool] = True
    tokenizer: Optional[str] = None
    join_method: Optional[str] = "union"


@dataclass
class Split:
    test_size: Optional[float] = 0.1
    seed: Optional[int] = 12345
    shuffle: Optional[bool] = True


@dataclass
class TokenizerComponent:
    type: str
    args: Dict[str, Any]


@dataclass
class TokenizerConfig:
    model: Dict[str, Any]
    vocab_size: Union[int, str] = "auto"
    batch_size: int = 1000
    normalizer: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    pre_tokenizer: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    decoder: Optional[Dict[str, Any]] = None
    trainer: Optional[Dict[str, Any]] = None
    use_sp_backend: Optional[bool] = False
    sp_kwargs: Optional[Dict[str, Any]] = None
    special_tokens: Dict[str, str] = field(
        default_factory=lambda: {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>"
        }
    )

    @staticmethod
    def convert_to_tokenizer_component(
            data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Union[TokenizerComponent, List[TokenizerComponent]]:

        if isinstance(data, list):
            return [TokenizerComponent(type=d.pop("type").lower(), args=d) for d in data]
        else:
            return TokenizerComponent(type=data.pop("type").lower(), args=data)

    def __post_init__(self):
        for attr in ["model", "trainer", "normalizer", "pre_tokenizer", "decoder"]:
            value = getattr(self, attr)
            if value:
                setattr(self, attr, self.convert_to_tokenizer_component(value))


@dataclass
class PeftConfig:
    # Defaults taken from MaLA-500:
    # https://arxiv.org/abs/2401.13303
    # https://github.com/MaLA-LM/mala-500/blob/faab723a9facab0a1eed1d55be60bbfd6876808e/continued_pretraining/continued_clm.py#L424
    task_type = "CAUSAL_LM"  # TODO: add MLM
    target_modules: Optional[List[str]] = field(default_factory=lambda: ["q_proj", "v_proj"])
    r: Optional[int] = 8
    lora_dropout: Optional[float] = 0.1
    lora_alpha: Optional[float] = 32.0
    bias: Optional[Union[str, float]] = "none"
    inference_mode: Optional[bool] = False


@dataclass
class TrainingParameters:
    model_type: str = "bert-base-uncased"
    task: str = "classification"
    num_train_epochs: int = 10
    max_length: int = 512
    batch_size: int = 8
    lr: float = 1e-3
    padding_strategy: str = "max_length"
    grad_accumulation_steps: int = 1
    mixed_precision: str = "no"
    mask_prob: Optional[float] = None
    num_eval_steps: Optional[int] = None
    quantize_4bit: Optional[bool] = False
    peft_config: Optional[Dict[str, Optional[Union[str, int, float, List[str]]]]] = None

    def __post_init__(self):
        if self.peft_config:
            self.peft_config = PeftConfig(**self.peft_config)


@dataclass
class Experiment:
    experiment_id: str = "default"
    wiki_id: str = "simple"
    local_path: Optional[str] = None
    hub_path: Optional[str] = None
    wandb_entity: Optional[str] = None
    experiment_folder: Optional[str] = None

    def __post_init__(self):
        if self.local_path:
            self.experiment_folder = f"{self.local_path}/{self.experiment_id}/{self.wiki_id}"


@dataclass
class SlurmAdditional:
    clusters: str
    account: str
    nodes: int
    cpus_per_gpu: int
    gpus_per_node: int
    mail_user: str 
    mail_type: str = "BEGIN,END,FAIL"


@dataclass
class Slurm:
    slurm_partition: str
    slurm_time: str  # with format 01:30:00 for 1.5 hours
    slurm_additional_parameters: Dict[str, Union[str, int]]

    def __post_init__(self):
        if self.slurm_additional_parameters:
            self.slurm_additional_parameters = SlurmAdditional(**self.slurm_additional_parameters)


@dataclass
class Dataset:
    export: bool = False
    push_to_hub: bool = False
    load_path: Optional[str] = None
    pre_filter: Optional[Dict[str, str]] = None
    partition: Optional[Dict[str, str]] = None
    split: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.pre_filter:
            self.pre_filter = PreFilter(**self.pre_filter)
        if self.partition:
            self.partition = Partition(**self.partition)
        if self.split:
            self.split = Split(**self.split)


@dataclass
class Tokenizer:
    export: bool = False
    push_to_hub: bool = False
    load_path: Optional[str] = None
    merge_with: Optional[str] = None
    tokenizer_config: Optional[Dict[str, Union[str, int, bool, Dict[str, str]]]] = None

    def __post_init__(self):
        if self.tokenizer_config:
            self.tokenizer_config = TokenizerConfig(**self.tokenizer_config)


@dataclass
class Pretrain:
    load_path: str
    export: bool = False
    push_to_hub: bool = False
    training_parameters: Optional[Dict[str, Union[str, int, float, bool]]] = None
    checkpoint: Optional[bool] = False
    test_path: Optional[str] = None

    def __post_init__(self):
        if self.training_parameters:
            self.training_parameters = TrainingParameters(**self.training_parameters)


@dataclass
class Finetune:
    load_path: str
    dataset_path: str
    export: bool = False
    push_to_hub: bool = False
    do_train: bool = False
    training_parameters: Optional[Dict[str, Union[str, int, float, bool]]] = None

    def __post_init__(self):
        if self.training_parameters:
            self.training_parameters = TrainingParameters(**self.training_parameters)


@dataclass
class MainConfig:
    experiment: Dict[str, Any]
    dataset: Optional[Dict[str, Any]] = None
    tokenizer: Optional[Dict[str, Any]] = None
    pretrain: Optional[Dict[str, Any]] = None
    finetune: Optional[Dict[str, Any]] = None
    slurm: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.experiment = Experiment(**self.experiment)
        if self.dataset:
            self.dataset = Dataset(**self.dataset)
        if self.tokenizer:
            self.tokenizer = Tokenizer(**self.tokenizer)
        if self.pretrain:
            self.pretrain = Pretrain(**self.pretrain)
        if self.finetune:
            self.finetune = Finetune(**self.finetune)
        if self.slurm:
            self.slurm = Slurm(**self.slurm)
