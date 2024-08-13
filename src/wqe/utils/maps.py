from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers
)

from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding
)

from ..data.metrics import *

PARTITION_MAP = {
    "length_chars": LengthCharacters,
    "length_words": LengthWords,
    "length_subwords": LengthSubwords,
    "unique_chars": UniqueCharacters,
    "unique_words": UniqueWords,
    "unique_subwords": UniqueSubwords,
    "unique_character_trigrams": UniqueCharacterTrigrams,
    "unique_trigrams": UniqueTrigrams,
    "unique_subword_trigrams": UniqueSubwordTrigrams,
    "alpha_chars": AlphaChars
}

TOKENIZER_PARAM_MAP = {
    "model": {
        "unigram": models.Unigram,
        "bpe": models.BPE,
        "wordpiece": models.WordPiece
    },
    "normalizer": {
        "nfc": normalizers.NFC,
        "nfd": normalizers.NFD,
        "nfkc": normalizers.NFKC,
        "nfkd": normalizers.NFKD,
        "nmt": normalizers.Nmt,
        "prepend": normalizers.Prepend,
        "replace": normalizers.Replace,
        "strip": normalizers.Strip
    },
    "pre_tokenizer": {
        "byte_level": pre_tokenizers.ByteLevel,
        "metaspace": pre_tokenizers.Metaspace,
        "whitespace": pre_tokenizers.Whitespace,
        "whitespace_split": pre_tokenizers.WhitespaceSplit,
        "unicode_scripts": pre_tokenizers.UnicodeScripts,
        "punctuation": pre_tokenizers.Punctuation,
        "digits": pre_tokenizers.Digits,
        "bert": pre_tokenizers.BertPreTokenizer,
        "split": pre_tokenizers.Split
    },
    "decoder": {
        "metaspace": decoders.Metaspace,
        "bpe": decoders.BPEDecoder,
        "wordpiece": decoders.WordPiece,
        "byte_level": decoders.ByteLevel
    },
    "trainer": {
        "unigram": trainers.UnigramTrainer,
        "bpe": trainers.BpeTrainer,
        "wordpiece": trainers.WordPieceTrainer
    },
}

TASK_TO_MODEL_AND_COLLATOR_MAPPING = {
    "mlm": {
        "model": AutoModelForMaskedLM,
        "collator": DataCollatorForLanguageModeling
    },
    "tagger": {
        "model": AutoModelForTokenClassification,
        "collator": DataCollatorForTokenClassification
    },
    "classifier": {
        "model": AutoModelForSequenceClassification,
        "collator": DataCollatorWithPadding
    }
}
