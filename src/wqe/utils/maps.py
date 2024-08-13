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
    "unique_words": UniqueWords,
    "unique_trigrams": UniqueTrigrams,
    "unique_chars": UniqueCharacters,
    "unique_char_trigrams": UniqueCharacterTrigrams,
    "unique_subwords": UniqueSubwords,
    "unique_subword_trigrams": UniqueSubwordTrigrams,
    "alpha_chars": AlphaChars,
    "frac_all_caps_words": FracAllCapsWords,
    "frac_lines_end_ellipsis": FracLinesEndEllipsis,
    "frac_no_alpha_words": FracNoAlphaWords,
    "doc_mean_word_length": MeanWordLength,
    "frac_unique_words": FracUniqueWords,
    "frac_unique_chars": FracUniqueChars,
    "unigram_entropy": UnigramEntropy,
    "trigram_entropy": TrigramEntropy,
    "lines_end_with_punctuation": LinesEndWithPunctuation,
    "num_lines": NumLines,
    "num_words_per_line": NumWordsPerLine
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
