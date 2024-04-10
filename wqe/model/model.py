import json
import torch

from collections import OrderedDict

from accelerate import Accelerator
from tqdm import tqdm
from transformers import DebertaConfig, RobertaConfig, BertConfig
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, DataCollatorForLanguageModeling
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW

from wqe.eval.eval import LossLogger

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
        self.num_train_epochs = None
        self.num_train_steps = None
        self.collator = None
        self.loaders = OrderedDict(
            train=None,
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
            remove_columns=dataset["train"].column_names,
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
        self.model, self.optimizer, self.loaders["train"], self.loaders["test"] = self.accelerator.prepare(
            self.model, self.optimizer, self.loaders["train"], self.loaders["test"]
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
        dev_loss_logger = LossLogger(
            len(self.loaders["test"]),
            increment_by=self.config["eval_steps"]
        )
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
                    # running_loss = 0.0
                    for dev_batch in self.loaders["test"]:
                        with torch.no_grad():
                            outputs = self.model(**dev_batch)
                            dev_loss_logger(outputs.loss.item())
                            # loss = outputs.loss
                            # running_loss += loss.item()
                    # dev_loss = running_loss / len(self.loaders["dev"])
                    progress_bar.set_description(f"Dev loss: {dev_loss_logger.get_metric()}")
                    self.model.train()

        dev_loss_logger.output_stats(self.config["export"]["path"] + "/dev_loss.txt")

    def save(self):
        export_config = self.config["export"]
        path = export_config["path"]

        if export_config["export_type"] == "hub":
            self.model.push_to_hub(
                path,
                data_dir=self.lang,
                use_temp_dir=True,
                repo_name=export_config["path"],
                private=True
                # organization=export_config["organization"],
                # commit_message=export_config["commit_message"]
            )

        elif export_config["export_type"] == "local":
            self.model.save_pretrained(path)

        else:
            raise ValueError("Invalid export type.")


class WikiNER(WikiModelFromConfig):
    def __init__(self, config):
        super().__init__(config)
        self.label_to_id = None
        self.id_to_label = None
        self.label_set = None
        self.tokenizer = None
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

    def _tokenize_and_align_labels(self, example):
        tokenized_input = self.tokenizer(
            example["tokens"],
            is_split_into_words=True
        )

        seen = set()
        labels = []
        for idx in tokenized_input.word_ids()[1:-1]:
            if idx in seen:
                labels.append(-100)
            else:
                labels.append(example['ner_tags'][idx])
                seen.add(idx)

        tokenized_input["labels"] = [-100] + labels + [-100]

        return tokenized_input

    def _prepare_data_for_training(self, dataset, tokenizer):

        self.tokenizer = tokenizer
        for k, v in self.tokenizer_kwargs.items():
            setattr(tokenizer, k, v)

        self.label_set = dataset["train"].features["ner_tags"].feature.names
        self.label_to_id = {label: i for i, label in enumerate(self.label_set)}
        self.id_to_label = {i: label for i, label in enumerate(self.label_set)}
        self.collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        batched_datasets = dataset.map(
            self._tokenize_and_align_labels,
            remove_columns=dataset["train"].column_names
        )

        for key in self.loaders:
            self.loaders[key] = DataLoader(
                batched_datasets[key],
                collate_fn=self.collator,
                batch_size=self.config["batch_size"]
            )

            if key == "train":
                self.loaders[key].shuffle = True

    def _init_training_parameters(self):

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config["from_pretrained"],
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )

        self.model = self.model.to("cuda")

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

    def _process_outputs_for_eval(self, outputs):

        predictions = outputs.logits.argmax(dim=-1)
        labels = outputs["labels"]

        # Necessary to pad predictions and labels for being gathered
        predictions = self.accelerator.pad_across_processes(
            predictions, dim=1, pad_index=-100
        )
        labels = self.accelerator.pad_across_processes(
            labels, dim=1, pad_index=-100
        )

        preds = self.accelerator.gather(predictions).detach().cpu().clone().numpy()
        labels = self.accelerator.gather(labels).detach().cpu().clone().numpy()

        true_labels = [l for label in labels for l in label if l != -100]
        true_preds = [
            p for prediction, label in zip(predictions, labels)
            for p, l in zip(prediction, label) if l != -100
        ]

        return true_labels, true_preds

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

            self.model.eval()
            running_loss = 0.0
            for dev_batch in self.loaders["dev"]:
                with torch.no_grad():
                    outputs = self.model(**dev_batch)
                true_preds, true_labels = self._process_outputs_for_eval(outputs)

            val_loss = running_loss / len(self.loaders["dev"])
            progress_bar.set_description(f"Validation loss: {val_loss}").refresh()
            self.model.train()
