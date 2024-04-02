import argparse
import yaml

from data.data import WikiDatasetFromConfig
from data.partition import Length
from tokenizer.tokenizer import WikiTokenizerFromConfig


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
    if config["data"]["pre_filter"]["do_pre_filter"]:
        dataset.pre_filter()
        if config["data"]["export"]["do_export"]:
            dataset.save()
    if config["data"]["partition"]["do_partition"]:
        dataset.apply_partition()
    tokenizer = WikiTokenizerFromConfig(config)


if __name__ == "__main__":
    main()
