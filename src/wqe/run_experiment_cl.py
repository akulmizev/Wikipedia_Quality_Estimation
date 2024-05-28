"""Command-line interface for running experiments.

Usage:
    run_experiment.py [ experiment [<experiment_id>]] [data] [tokenizer] [pretrain] [finetune]
    run_experiment.py (-h | --help)

Options:
    -h --help   Show this screen.

"""


import argparse
import os
import yaml

import click

# from datasets import load_dataset
# from huggingface_hub import HfApi
#
# from data.data import WikiLoader
from utils.config import *

# from tokenizer.tokenizer import PreTrainedTokenizerFast
#
# from model.pretrain import MLM
# from model.finetune import Tagger, Classifier

@click.group()
def cli():
    """Command-line interface for running experiments."""

@cli.group()
def experiment():
    """Run experiment."""


@cli.group()
def dataset():
    """Run experiment."""

@experiment.command("experiment")
@click.option("--wiki_id")
@click.option("--experiment_id")
@click.option("--wandb_entity")
@click.option("--local_path")
@click.option("--hub_path")
def parse_experiment(experiment_id, wandb_entity, local_path, hub_path, wiki_id):
    experiment_cfg = Experiment(
        experiment_id=experiment_id,
        wandb_entity=wandb_entity,
        local_path=local_path,
        hub_path=hub_path,
        wiki_id=wiki_id
    )

    return experiment_cfg

@dataset.command("dataset")
@click.argument("dataset_load")
@click.option("--dataset_export")
@click.option("--dataset_push_to_hub")
@click.option("--pre_filter")
@click.option("--partition")
@click.option("--split")
def parse_dataset(dataset_load, dataset_export, dataset_push_to_hub, pre_filter, partition, split):
    dataset_cfg = Dataset(
        dataset_load=dataset_load,
        dataset_export=dataset_export,
        dataset_push_to_hub=dataset_push_to_hub,
        pre_filter=pre_filter,
        partition=partition,
        split=split
    )

def main():

    experiment_cfg = parse_experiment()

    print("SHIT")

    # parser = argparse.ArgumentParser()
    # subparsers = parser.add_subparsers(help='types of A')
    #
    # experiment = subparsers.add_parser("experiment")
    # dataset = subparsers.add_parser("dataset")
    #
    # experiment.add_argument("--experiment_id", type=str, help="Specify experiment ID.")
    # experiment.add_argument("--wandb_entity", type=str, help="Specify entity for Weights & Biases.")
    # experiment.add_argument("--local_path", type=str, help="Specify local path for saving experiment data.")
    # experiment.add_argument("--hub_path", type=str, help="Specify hub path for saving experiment data.")
    #
    # dataset.add_argument("--dataset_load", type=str, help="Specify dataset load.")
    # dataset.add_argument("--dataset_export", type=bool, help="Specify dataset export.")
    # dataset.add_argument("--dataset_push_to_hub", type=bool, help="Specify dataset push to hub.")
    # dataset.add_argument("--pre_filter", type=str, help="Specify pre filter.")
    # dataset.add_argument("--partition", type=str, help="Specify partition.")
    # dataset.add_argument("--split", type=str, help="Specify split.")



        # model.train(dataset)
        # model.save(f"{experiment_path}/model/{experiment_cfg.wiki_id}")
        # if api.repo_exists(f"{experiment_cfg.hub_path}/{experiment_cfg.experiment_id}"):
        #     api.create_repo(
        #         repo_id=f"{experiment_cfg.hub_path}/{experiment_cfg.experiment_id}",
        #         repo_type="model",
        #         private=False
        #     )
        # api.upload_folder(
        #     folder_path=f"{experiment_path}/model/",
        #     repo_id=f"{experiment_cfg.hub_path}/{experiment_cfg.experiment_id}",
        #     repo_type="model"
        # )
        #     if config["pretrain"]["train"]:
        #         model.train(dataset)
        #     if "test_data" in config["pretrain"]:
        #         test_dataset = WikiDatasetFromConfig.load_dataset_directly(
        #             config["pretrain"]["test_data"],
        #             wiki_id=args.wiki_id, split="test"
        #         )
        #         model.test(test_dataset)


        # elif "from_pretrained" in tokenizer_cfg:
        #     tokenizer = FastTokenizerFromConfig.from_pretrained(tokenizer_cfg["from_pretrained"])
    # if "tokenizer" in config:
    #     if "from_config" in config["tokenizer"]:
    #         tokenizer = FastTokenizerFromConfig.train_from_config(
    #             dataset["train"],
    #             config_path=config["tokenizer"]["from_config"],
    #             batch_size=1000)
    #         if "export" in config["tokenizer"]:
    #             tokenizer.save()
    #     elif "from_pretrained" in config["tokenizer"]:
    #         tokenizer = FastTokenizerFromConfig.from_pretrained(config["tokenizer"]["from_pretrained"])
    #
    # if "pretrain" in config:
    #     model = WikiMLM(config, tokenizer, dataset)
    #     if config["pretrain"]["train"]:
    #         model.train(dataset)
    #     if "test_data" in config["pretrain"]:
    #         test_dataset = WikiDatasetFromConfig.load_dataset_directly(
    #             config["pretrain"]["test_data"],
    #             wiki_id=args.wiki_id, split="test"
    #         )
    #         model.test(test_dataset)

    # if "finetune" in config:
    #     ner_dataset = load_dataset("indic_glue", f"wiki-ner.{config['wiki_id']}")
    #     model = WikiNER(config["finetune"])
    #     model.prepare_model(ner_dataset, fast_tokenizer)
    #     model.train()

    pass


if __name__ == "__main__":
    main()