import json
import torch

from collections import OrderedDict

from accelerate import Accelerator
from datasets import load_dataset
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from transformers import DebertaConfig, RobertaConfig, BertConfig
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, DataCollatorForLanguageModeling
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW

CONFIG_MAPPING = {
    "deberta": DebertaConfig,
    "roberta": RobertaConfig,
    "bert": BertConfig
}

TASK_MAPPING = {
    "mlm": {
        "model": AutoModelForMaskedLM,
        "collator": DataCollatorForLanguageModeling
    },
    "ner": {
        "model": AutoModelForTokenClassification,
        "collator": DataCollatorForTokenClassification
    }
}

class WikiModelFromConfig:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.accelerator = None
        self.scheduler = None
        self.num_train_epochs = self.config["num_train_epochs"]
        self.num_train_steps = None
        self.collator = None
        self.loaders = OrderedDict(
            train=None,
            dev=None,
            test=None
        )

    def _get_model_config(self):

        model_config = CONFIG_MAPPING.get(self.config["model_type"])
        model_config = model_config.from_json_file(self.config["from_config"])

        return model_config

    def _prepare_data_for_training(self, dataset, tokenizer):
        raise NotImplementedError("Subclasses must implement this method.")

    def _init_training_parameters(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def prepare_model(self, dataset, tokenizer):
        self._prepare_data_for_training(dataset, tokenizer)
        self._init_training_parameters()

    def train(self):
        raise NotImplementedError("Subclasses must implement this method.")


class WikiMLM(WikiModelFromConfig):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer_kwargs = OrderedDict(
            mask_token="[MASK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="[CLS]",
            unk_id=0
        )

    def _prepare_data_for_training(self, dataset, tokenizer):

        for k, v in self.tokenizer_kwargs.items():
            setattr(tokenizer, k, v)

        self.collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=self.config["mask_prob"]
        )

        batched_datasets = dataset.map(
            lambda examples: tokenizer(
                examples["text"],
                max_length=self.config["max_length"],
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True),
            batched=True,
            remove_columns=["text", "url", "title", "id"],
        )

        # DeBERTa doesn't accept the overflow_to_sample_mapping column, so deleting it.
        batched_datasets = batched_datasets.remove_columns("overflow_to_sample_mapping")

        for key in self.loaders:
            self.loaders[key] = DataLoader(
                batched_datasets[key],
                collate_fn=self.collator,
                batch_size=self.config["batch_size"]
            )

            if key == "train":
                self.loaders[key].shuffle = True

    def _init_training_parameters(self):

        if "from_config" in self.config:
            self.model = AutoModelForMaskedLM.from_config(self._get_model_config()).to("cuda")
        elif "from_pretrained" in self.config:
            self.model = AutoModelForMaskedLM.from_pretrained(self.config["from_pretrained"]).to("cuda")
        else:
            raise ValueError("`from_config` or `from_pretrained` must be in the configuration.")

        self.optimizer = AdamW(self.model.parameters(), lr=float(self.config["lr"]))
        self.accelerator = Accelerator()
        self.model, self.optimizer, self.loaders["train"], self.loaders["dev"], self.loaders["test"] = self.accelerator.prepare(
            self.model, self.optimizer, self.loaders["train"], self.loaders["dev"], self.loaders["test"]
        )

        self.num_train_steps = self.num_train_epochs * len(self.loaders["train"])
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_train_steps
        )

    def prepare_model(self, dataset, tokenizer):

        self._prepare_data_for_training(dataset, tokenizer)
        self._init_training_parameters()

    def train(self):
        progress_bar = tqdm(range(self.num_train_steps))
        for epoch in range(self.num_train_epochs):
            self.model.train()
            for i, batch in enumerate(self.loaders["train"]):
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

                if i > 0 and i % self.config["eval_steps"] == 0:
                    self.model.eval()
                    running_loss = 0.0
                    for batch in self.loaders["dev"]:
                        with torch.no_grad():
                            outputs = self.model(**batch)
                            loss = outputs.loss
                            running_loss += loss.item()
                    val_loss = running_loss / len(self.loaders["dev"])
                    progress_bar.set_description(f"Validation loss: {val_loss}").refresh()
                    self.model.train()

        self.model.eval()
        running_loss = 0.0
        for batch in self.loaders["test"]:
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                running_loss += loss.item()
        print(f"Test loss: {running_loss / len(self.loaders['test'])}")

        if "export" in self.config:
            self.model.save_pretrained(self.config["export"]["path"])

            with open(self.config["export"]["path"] + "/config.json", "w") as f:
                json.dump(self.model.config.to_dict(), f)

