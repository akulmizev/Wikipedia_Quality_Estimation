from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

PARAM_MAP = {
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
