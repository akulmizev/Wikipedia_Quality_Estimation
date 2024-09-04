from datasets import concatenate_datasets

from wqe import WikiLoader, HfSentencePieceTokenizer, GOPHER_THRESHOLDS
from wqe.utils.config import TokenizerConfig

WIKI_ID = "ha"
HUB_PATH = "WikiQuality"

ALL_MASAKHANE_LANGS = [
    'sw', 'ha', 'yo', 'ig', 'am',
    'sn', 'zu', 'ary', 'so', 'rw',
    'tw', 'ln', 'lg', 'xh', 'wo',
    'om', 'tn', 'ny', 'pcm', 'ee',
    'bm', 'ts', 'rn', 'fon', 'ti'
]

METRICS = [
    "length_chars",
    "unique_words",
    "unique_trigrams",
    "frac_unique_words",
    "frac_unique_trigrams",
    "unigram_entropy",
    "trigram_entropy"
]

TOKENIZER_CONFIG = TokenizerConfig(
    model={"type": "unigram"},
    vocab_size="auto"
)


def make_datasets(wiki_id: str) -> None:

    raw_wiki = WikiLoader(wiki_id).load_dataset().generate_splits(
        test_size=0.05,
        shuffle=True,
        seed=42
    )
    raw_wiki.push_to_hub(f"{HUB_PATH}/raw_wiki", wiki_id)

    tokenizer = HfSentencePieceTokenizer.train_from_config(raw_wiki["train"]["text"], TOKENIZER_CONFIG)
    tokenizer.push_to_hub(f"{HUB_PATH}/raw_wiki.{wiki_id}")

    pre_filtered = WikiLoader.from_dataset(raw_wiki["train"], wiki_id)
    pre_filtered = pre_filtered.pre_filter(script_regex=True)
    pre_filtered = pre_filtered.deduplicate(
        exact_match=True,
        min_hash=True,
        jaccard_threshold=0.85,
        n_shingles=3
    )
    pre_filtered.generate_splits(
        test_size=0.05,
        shuffle=True,
        seed=42
    )
    pre_filtered.push_to_hub(f"{HUB_PATH}/pre_filtered", wiki_id)

    tokenizer = HfSentencePieceTokenizer.train_from_config(pre_filtered["train"]["text"], TOKENIZER_CONFIG)
    tokenizer.push_to_hub(f"{HUB_PATH}/pre_filtered.{wiki_id}")

    pre_filtered_concat = concatenate_datasets([pre_filtered["train"], pre_filtered["test"]])

    thresholded = pre_filtered.apply_threshold(GOPHER_THRESHOLDS, keep_columns=False)
    thresholded.push_to_hub(f"{HUB_PATH}/gopher", wiki_id)

    # Make high/low quality partitions for each partitioning method
    for metric in METRICS:
        for quality in [True, False]:
            pre_filtered_train = WikiLoader.from_dataset(pre_filtered_concat, wiki_id)
            partitioned = pre_filtered_train.apply_partition(
                split_method="balanced_chars",
                metrics=metric,
                quality=quality,
                keep_columns=False
            )
            partitioned.generate_splits(
                test_size=0.05,
                shuffle=True,
                seed=42
            )

            quality_id = "hi" if quality else "lo"
            partitioned.push_to_hub(f"{HUB_PATH}/{metric}_{quality_id}", wiki_id)


if __name__ == "__main__":
    for lang in ALL_MASAKHANE_LANGS:
        make_datasets(lang)
