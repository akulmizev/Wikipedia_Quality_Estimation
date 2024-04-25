import argparse
import os
import yaml

from huggingface_hub import HfApi

from data.data import WikiLoader
from utils.config import parse_config

from tokenizer.tokenizer import FastTokenizerFromConfig
# from model.model import WikiMLM
# from model.model import WikiNER

def main():
    global experiment_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str, required=True,
                        help="Specify path to configuration file.")
    parser.add_argument("--wiki_id", default=None, type=str, required=True,
                        help="Specify Wiki ID for extracting Wiki dump.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        config["experiment"].update({"wiki_id": args.wiki_id})

    experiment_cfg, data_cfg, tokenizer_cfg, pretrain_cfg = parse_config(config)

    if experiment_cfg.local_path:
        experiment_path = f"{experiment_cfg.local_path}/{experiment_cfg.experiment_id}"
        os.makedirs(experiment_path, exist_ok=True)

    if experiment_cfg.hub_path:
        api = HfApi()

    if data_cfg:
        dataset = WikiLoader(data_cfg, wiki_id=experiment_cfg.wiki_id)
        if data_cfg.pre_filter:
            dataset.do_pre_filter()
        if data_cfg.partition:
            dataset.apply_partition()
        if data_cfg.split:
            dataset.generate_splits()
        if data_cfg.export:
            dataset.save(f"{experiment_path}/data/{experiment_cfg.wiki_id}")
            if api.repo_exists(f"{experiment_cfg.hub_path}/{experiment_cfg.experiment_id}"):
                api.create_repo(
                    repo_id=f"{experiment_cfg.hub_path}/{experiment_cfg.experiment_id}",
                    repo_type="dataset",
                    private=False
                )
            api.upload_folder(
                folder_path=f"{experiment_path}/data/",
                repo_id=f"{experiment_cfg.hub_path}/{experiment_cfg.experiment_id}",
                repo_type="dataset"
            )

    if tokenizer_cfg:
        if tokenizer_cfg.load.method == "config":
            tokenizer = FastTokenizerFromConfig.train_from_config(
                dataset["train"],
                config=tokenizer_cfg.parameters,
                batch_size=1000
            )
            if tokenizer_cfg.export:
                tokenizer.save_pretrained(f"{experiment_path}/model/{experiment_cfg.wiki_id}")
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
