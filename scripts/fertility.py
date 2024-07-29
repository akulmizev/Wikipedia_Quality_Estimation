import os
import yaml

from collections import Counter
from typing import List, Dict

import numpy as np
import pandas as pd


from wqe import WikiLoader, HfTokenizerFromConfig, validate_and_format_dataset

from wqe.utils.config import Tokenizer

TO_IGNORE = ["Ġ", "▁", "##", "Ċ"]
CONFIG_DIR = "../scratch/tokenizer_configs"
LANGS = [
    'ha', 'yo', 'ig', 'sn',
    'zu', 'rw', 'tw', 'lg',
    'wo', 'tn', 'ny', "sw"
    'pcm', 'ee', 'bm', 'fon'
]


def load_dataset(lang: str) -> Dict[str, List[str]]:
    """
    Load the Wikipedia dataset and NER dataset for the given language.

    Args:
        lang (str): The language code.

    Returns:
        Dict[str, List[str]]: A dictionary containing the Wikipedia dataset and the NER dataset.
    """
    if lang == "xh":
        wiki = WikiLoader(lang).load_dataset().pre_filter(char_cutoff=100)
    else:
        wiki = WikiLoader(lang).load_dataset().pre_filter(script_regex=True, char_cutoff=100)

    ner_dataset = validate_and_format_dataset("masakhane/masakhaner2", lang, "ner")["validation"]

    return {"wiki": wiki["train"]["text"], "ner": ner_dataset}


def calculate_tokenization_stats(dataset: Dict[str, List[str]], tokenizer) -> Dict[str, float]:
    """
    Calculate tokenization statistics for the given dataset and tokenization.

    Args:
        dataset (Dict[str, List[str]]): A dictionary containing the Wikipedia dataset and the NER dataset.
        tokenizer: The tokenization object.

    Returns:
        Dict[str, float]: A dictionary containing tokenization statistics.
    """
    fertilities = []
    num_standalone_meta_tokens = []
    for example in dataset["ner"]:
        tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
        fertilities.extend(Counter(tokenized_input.word_ids()).values())
        meta_token_counter = 0
        for i in tokenized_input["input_ids"]:
            if tokenizer.convert_ids_to_tokens(i) in TO_IGNORE:
                meta_token_counter += 1
        num_standalone_meta_tokens.append(meta_token_counter)
    return {
        "mean_fertility": np.mean(fertilities),
        "std_fertility": np.std(fertilities),
        "meta_tokens_per_sentence": np.mean(num_standalone_meta_tokens),
        "vocab_size": len(tokenizer.get_vocab()),
    }


def main():
    """
    Main function to analyze tokenization for various languages and configurations.
    """
    stats = []
    for lang in LANGS:
        print(f"Processing language: {lang}")
        dataset = load_dataset(lang)
        for config_file in os.listdir(CONFIG_DIR):
            config_name = config_file.split(".")[0]
            with open(CONFIG_DIR + config_file, "r") as f:
                raw_config = yaml.safe_load(f)
            config = Tokenizer(**raw_config)
            tokenizer_config = config.tokenizer_config
            tokenizer = HfTokenizerFromConfig.train_from_config(dataset["wiki"], tokenizer_config)
            stats.append({
                "lang": lang,
                "config": config_name,
                **calculate_tokenization_stats(dataset, tokenizer)
            })

            df = pd.DataFrame(
                stats,
                columns=["lang", "config", "mean_fertility", "std_fertility", "meta_tokens_per_sentence", "vocab_size"]
            )
            df.to_csv("tokenizer_fertility.csv", index=False, float_format="%.4f")

    df = pd.DataFrame(
        stats,
        columns=["lang", "config", "mean_fertility", "std_fertility", "meta_tokens_per_sentence", "vocab_size"]
    )
    df.to_csv("tokenizer_fertility.csv", index=False, float_format="%.4f")
    print("Analysis completed. Results saved to tokenizer_fertility.csv")


if __name__ == "__main__":
    main()
