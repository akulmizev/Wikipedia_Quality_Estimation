import argparse
import yaml

from data.data import WikiDatasetFromConfig
from tokenizer.tokenizer import WikiTokenizerFromConfig
from model.model import WikiMLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str, required=True,
                        help="Specify path to configuration file.")
    parser.add_argument("--wiki_id", default=None, type=str, required=True,
                        help="Specify Wiki ID for extracting Wiki dump.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        config.update({"wiki_id": args.wiki_id})

    dataset = WikiDatasetFromConfig(config)
    if "pre_filter" in config["data"]:
        dataset.pre_filter()
    if "split" in config["data"]:
        dataset.generate_splits()
    if "partition" in config["data"]:
        dataset.apply_partition()
    if "export" in config["data"]:
        dataset.save()

    tokenizer = WikiTokenizerFromConfig(config)
    if "train" in config["tokenizer"]:
        tokenizer.train(dataset["train"], batch_size=100)
        if "export" in config["tokenizer"]:
            tokenizer.save()
    fast_tokenizer = tokenizer.convert_to_fast()

    model = WikiMLM(config["pretrain"])
    model.prepare_model(dataset, fast_tokenizer)
    model.train()

    pass

if __name__ == "__main__":
    main()
