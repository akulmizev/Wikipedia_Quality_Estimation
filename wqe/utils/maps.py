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

# from ..data.partition import (
from data.partition import (
    Length,
    UniqueSubwords,
    UniqueSubwordTrigrams,
    UniqueTrigrams,
    UniqueWords,
    UniqueCharacters,
    UniqueCharacterTrigrams,
    AlphaChars
)

PARTITION_MAP = {
    "length": Length,
    "unique_subwords": UniqueSubwords,
    "unique_subword_trigrams": UniqueSubwordTrigrams,
    "unique_trigrams": UniqueTrigrams,
    "unique_words": UniqueWords,
    "unique_characters": UniqueCharacters,
    "unique_character_trigrams": UniqueCharacterTrigrams,
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
        "nfkd": normalizers.NFKD
    },
    "pre_tokenizer": {
        "byte_level": pre_tokenizers.ByteLevel,
        "metaspace": pre_tokenizers.Metaspace,
        "whitespace": pre_tokenizers.Whitespace,
        "unicode_scripts": pre_tokenizers.UnicodeScripts,
        "digits": pre_tokenizers.Digits
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
    "ner": {
        "model": AutoModelForTokenClassification,
        "collator": DataCollatorForTokenClassification
    },
    "sentiment_analysis": {
        "model": AutoModelForSequenceClassification,
        "collator": DataCollatorWithPadding
    }
}
