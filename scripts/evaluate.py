import argparse
from transformers import AutoModelForMaskedLM, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import math

import accelerate

def argument_parser():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument("--model", type=str, required=True, help="The model to evaluate.")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset to evaluate on.")
    return parser.parse_args()

class ModelEval():
    def __init__(self, model, tokenizer, dataset):
        self.model = AutoModelForMaskedLM.from_pretrained(model)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer)
        self.dataset = load_dataset(dataset, split="test")
        self.accelerator = accelerate.Accelerator(device_placement=True, mixed_precision="bf16")
        self.padding_strategy = "longest"
        self.max_length = 512
        self.batch_size = 32
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.4,
            pad_to_multiple_of=8
        )
        self.model.eval()

    def tokenize_and_collate(self, dataset):
        with self.accelerator.main_process_first():
            batched_dataset = dataset.map(
                lambda examples: self.tokenizer(
                    examples["text"],
                    padding=self.padding_strategy,
                    max_length=self.max_length,
                    truncation=True,
                    return_overflowing_tokens=True),
                batched=True,
                remove_columns=dataset.column_names
            )
        batched_dataset = batched_dataset.remove_columns("overflow_to_sample_mapping")

        loader = DataLoader(
            batched_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )
        return loader

    def eval_loop(self, loader):
        running_loss = []
        for batch in loader:
            with torch.no_grad():
                outputs = self.model(**batch)
            loss = outputs.loss
            running_loss.append(loss.item())
        eval_loss = math.fsum(running_loss) / len(running_loss)
        perplexity = math.exp(eval_loss)

        return {"loss": eval_loss, "perplexity": perplexity}


def main():
    args = argument_parser()
    model_eval = ModelEval(args.model, args.model, args.dataset)
    loader = model_eval.tokenize_and_collate(model_eval.dataset)
    metrics = model_eval.eval_loop(loader)
    print(metrics)



