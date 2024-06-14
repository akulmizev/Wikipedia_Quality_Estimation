import sys

from wqe import WikiLoader, FastTokenizerFromConfig

from wqe.utils.config import TokenizerConfig

EXPERIMENT_DIR = sys.argv[1]
HUB_PATH = sys.argv[2]

LANGS = ["ha", "yo", "ig", "sw", "pcm"]

ALL_LANGS = [
    'sw', 'ha', 'yo', 'ig', 'am',
    'sn', 'zu', 'ary', 'so', 'rw',
    'tw', 'ln', 'lg', 'xh', 'wo',
    'om', 'tn', 'ny', 'pcm', 'ee',
    'bm', 'ts', 'rn', 'fon', 'ti'
]

PARTITIONS = [
    "length",
    "unique_trigrams",
    "unique_words",
    "unique_character_trigrams"
]


def make_datasets(wiki_id: str) -> None:

    raw_wiki = WikiLoader(wiki_id).load_dataset().generate_splits(
        test_size=0.05,
        shuffle=True,
        seed=42
    )

    raw_wiki.save(f"{EXPERIMENT_DIR}/raw_wiki/{wiki_id}/data")
    raw_wiki.push_to_hub(f"{HUB_PATH}/raw_wiki", wiki_id)

    raw_wiki_train = WikiLoader.from_dataset(raw_wiki["train"], wiki_id)
    pre_filtered = raw_wiki_train.pre_filter(
        script_regex=True,
        lang_id=False,
        char_cutoff=100
    )
    pre_filtered.generate_splits(
        test_size=0.05,
        shuffle=True,
        seed=42
    )

    pre_filtered.save(f"{EXPERIMENT_DIR}/pre_filtered/{wiki_id}/data")
    pre_filtered.push_to_hub(f"{HUB_PATH}/pre_filtered", wiki_id)

    tokenizer_config = TokenizerConfig(
        model={"type": "unigram"},
        trainer={"type": "unigram"},
        normalizer={"type": "nfkc"},
        pre_tokenizer=[
            {"type": "digits", "individual_digits": True},
            {"type": "metaspace"}
        ],
        decoder={"type": "metaspace"},
        vocab_size="auto"
    )
    tokenizer = FastTokenizerFromConfig.train_from_config(pre_filtered["train"]["text"], tokenizer_config)
    tokenizer.save_pretrained(f"{EXPERIMENT_DIR}/pre_filtered/{wiki_id}/model")
    tokenizer.push_to_hub(f"{HUB_PATH}/pre_filtered.{wiki_id}")

    # Make high/low quality partitions for each partitioning method
    for partition in PARTITIONS:
        for quality in [True, False]:
            pre_filtered_train = WikiLoader.from_dataset(pre_filtered["train"], wiki_id)
            partitioned = pre_filtered_train.apply_partition(partition, quality=quality)
            partitioned.generate_splits(
                test_size=0.05,
                shuffle=True,
                seed=42
            )

            qual_id = "hi" if quality else "lo"

            partitioned.save(f"{EXPERIMENT_DIR}/{partition}_{qual_id}/{wiki_id}/data")
            partitioned.push_to_hub(f"{HUB_PATH}/{partition}_{qual_id}", wiki_id)

    # Make high/low quality partitions for combined methods
    for quality in [True, False]:
        pre_filtered_train = WikiLoader.from_dataset(pre_filtered["train"], wiki_id)
        partitioned = pre_filtered_train.apply_partition(PARTITIONS, quality=quality)
        partitioned.generate_splits(
            test_size=0.05,
            shuffle=True,
            seed=42
        )

        qual_id = "hi" if quality else "lo"

        partitioned.save(f"{EXPERIMENT_DIR}/all_methods_{qual_id}/{wiki_id}/data")
        partitioned.push_to_hub(f"{HUB_PATH}/all_methods_{qual_id}", wiki_id)


if __name__ == "__main__":

    for lang in LANGS:
        make_datasets(lang)
