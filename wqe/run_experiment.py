import argparse
import yaml

from dataclasses import dataclass

from data.data import WikiDatasetFromConfig
from tokenizer.tokenizer import WikiTokenizerFromConfig, WikiTokenizerFast
from model.model import WikiMLM
# from model.model import WikiNER

@dataclass
class DatasetConfig:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str, required=True,
                        help="Specify path to configuration file.")
    parser.add_argument("--wiki_id", default=None, type=str, required=True,
                        help="Specify Wiki ID for extracting Wiki dump.")
    args = parser.parse_args()

    # TODO: Make config checker to ensure all necessary fields are present
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        config.update({"wiki_id": args.wiki_id})

    if "data" in config:
        dataset = WikiDatasetFromConfig(config)
        if "pre_filter" in config["data"]:
            dataset.pre_filter()
        if "partition" in config["data"]:
            dataset.apply_partition()
        if "split" in config["data"]:
            dataset.generate_splits()
        if "export" in config["data"]:
            dataset.save()

    if "tokenizer" in config:
        if "from_config" in config["tokenizer"]:
            tokenizer = WikiTokenizerFast(config=config)
            tokenizer.train(dataset["train"], batch_size=1000)
            if "export" in config["tokenizer"]:
                tokenizer.save()
        elif "from_pretrained" in config["tokenizer"]:
            tokenizer = WikiTokenizerFast.from_pretrained(config["tokenizer"]["from_pretrained"])

    if "pretrain" in config:
        model = WikiMLM(config, tokenizer, dataset)
        if config["pretrain"]["train"]:
            model.train(dataset)
        if "test_data" in config["pretrain"]:
            test_dataset = WikiDatasetFromConfig.load_dataset_directly(
                config["pretrain"]["test_data"],
                wiki_id=args.wiki_id, split="test"
            )
            model.test(test_dataset)

    # if "finetune" in config:
    #     ner_dataset = load_dataset("indic_glue", f"wiki-ner.{config['wiki_id']}")
    #     model = WikiNER(config["finetune"])
    #     model.prepare_model(ner_dataset, fast_tokenizer)
    #     model.train()

    pass


if __name__ == "__main__":
    main()
