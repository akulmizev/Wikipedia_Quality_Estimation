import logging
import tempfile

from typing import Dict, List, Tuple, Union

from sentencepiece import SentencePieceProcessor
from transformers.utils.sentencepiece_model_pb2_new import ModelProto
# from sentencepiece.sentencepiece_model_pb2 import ModelProto
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE, Unigram
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def _get_prepend_scheme(
        add_prefix_space: bool,
        original_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> str:

    """
    Copied almost verbatim from: transformers.convert_slow_tokenizer
    Get the prepend scheme for the pre-tokenizer based on the `add_prefix_space` attribute of the original tokenizer.

    Parameters
    ----------

    add_prefix_space : bool
        Whether to add a prefix space to the tokenized text.

    original_tokenizer : PreTrainedTokenizer
        The original tokenizer.

    Returns
    -------

    str
    """

    if add_prefix_space:
        prepend_scheme = "always"
        if not getattr(original_tokenizer, "legacy", True):
            prepend_scheme = "first"
    else:
        prepend_scheme = "never"
    return prepend_scheme


def has_sentencepiece_backend(
        hf_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
):
    """
    Naively checks if a pretrained tokenizer has a SentencePiece backend via the `.vocab_file` attribute.

    Parameters
    ----------

    hf_tokenizer : Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        The tokenizer to check.

    Returns
    -------

    bool
    """

    vocab_file = hf_tokenizer.vocab_files_names.get("vocab_file", None)

    if vocab_file is None:
        return False
    else:
        return vocab_file.endswith(".model")


def merge_tokenizers(
        base_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        tokenizer_to_merge: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        include_special_tokens: bool = True
):

    """
    Merges the vocabulary of two tokenizers together, based on their set difference, e.g.:

    unique_vocab = tokenizer_to_merge.vocab - base_tokenizer.vocab

    Does not merge if the tokenizer algorithm (`bpe` or `unigram`) of both tokenizers does not match,
    as this can lead to unexpected tokenization results.

    If the backend tokenizer implementation does not match (e.g. `sentencepiece` or `tokenizers`),
    vocabulary is merged via the `Tokenizer.add_tokens()` method.
    This can also lead to unexpected tokenization results.

    If both tokenizers do have a sentencepiece backend, the vocabularies are merged by adding the unique tokens
    from `tokenizer_to_merge` to `base_tokenizer` via the suggested procedure in the SentencePiece documentation:

    https://github.com/google/sentencepiece/blob/master/python/add_new_vocab.ipynb

    Parameters
    ----------

    base_tokenizer : Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        The tokenizer to merge into.

    tokenizer_to_merge : Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        The tokenizer to merge from.

    include_special_tokens : bool
        Whether to include special tokens from `tokenizer_to_merge` in the merged tokenizer.

    Returns
    -------

    merged_tokenizer : Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    """

    if not isinstance(tokenizer_to_merge.backend_tokenizer.model, type(base_tokenizer.backend_tokenizer.model)):
        raise ValueError(
            "Both tokenizers must have the same backend model. ",
            "Received: ",
            type(base_tokenizer.backend_tokenizer.model),
            type(tokenizer_to_merge.backend_tokenizer.model),
        )

    if not has_sentencepiece_backend(base_tokenizer) or not has_sentencepiece_backend(tokenizer_to_merge):

        logger.warning(
            "Merging tokenizers with non-SentencePiece backends. "
            "May result in unexpected tokenization."
        )

        base_tokenizer_vocab = set(list(base_tokenizer.get_vocab().keys()))
        tokenizer_to_merge_vocab = set(list(tokenizer_to_merge.get_vocab().keys()))

        tokens_to_add = list(tokenizer_to_merge_vocab - base_tokenizer_vocab)

        base_tokenizer.add_tokens(tokens_to_add)
        if include_special_tokens:
            base_tokenizer.add_special_tokens({
                k: v for k, v in tokenizer_to_merge.special_tokens_map.items()
                if k not in base_tokenizer.special_tokens_map
            })

        return base_tokenizer

    else:
        base_tokenizer_sp = ModelProto()
        base_tokenizer_sp.ParseFromString(open(base_tokenizer.vocab_file, "rb").read())

        tokenizer_to_merge_sp = ModelProto()
        tokenizer_to_merge_sp.ParseFromString(open(tokenizer_to_merge.vocab_file, "rb").read())

        uniques = {piece.piece for piece in base_tokenizer_sp.pieces}
        for new_piece in tokenizer_to_merge_sp.pieces:
            if new_piece.piece not in uniques:
                piece_to_add = ModelProto().SentencePiece()
                piece_to_add.piece = new_piece.piece
                piece_to_add.score = 0
                base_tokenizer_sp.pieces.append(piece_to_add)

        logger.info(
            f"Merged {len(uniques)} new tokens into the base tokenizer. "
            f"New vocab size: {len(base_tokenizer_sp.pieces)}"
        )

        with tempfile.NamedTemporaryFile(suffix='.model') as temp_file:
            temp_file.write(base_tokenizer_sp.SerializeToString())
            temp_file_path = temp_file.name
            merged_tokenizer = AutoTokenizer.from_pretrained(
                base_tokenizer.name_or_path,
                vocab_file=temp_file_path,
                tokenizer_file=None
            )

        if include_special_tokens:
            merged_tokenizer.add_special_tokens({
                k: v for k, v in tokenizer_to_merge.special_tokens_map.items()
                if k not in merged_tokenizer.special_tokens_map
            })

        return merged_tokenizer


class SentencePieceExtractor:

    """
    Copied almost verbatim from: transformers.convert_slow_tokenizer
    Extractor implementation for SentencePiece trained models. https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):

        self.sp = SentencePieceProcessor(model_proto=model)

    def extract(self, vocab_scores=None) -> Tuple[Dict[str, int], List[Tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}

        if vocab_scores is not None:
            vocab_scores, reverse = dict(vocab_scores), True
        else:
            vocab_scores, reverse = vocab, False

        # Merges
        merges = []
        for merge, piece_score in vocab_scores.items():
            local = []
            for index in range(1, len(merge)):
                piece_l, piece_r = merge[:index], merge[index:]
                if piece_l in vocab and piece_r in vocab:
                    local.append((piece_l, piece_r, piece_score))
            local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]))
            merges.extend(local)

        merges = sorted(merges, key=lambda val: val[2], reverse=reverse)
        merges = [(val[0], val[1]) for val in merges]
        return vocab, merges


class SpmConverter:

    """
    Copied almost verbatim from: transformers.convert_slow_tokenizer
    Converts a SentencePiece tokenizer to a `tokenizers.Tokenizer`.
    """

    def __init__(
            self,
            original_tokenizer: PreTrainedTokenizer
    ):

        if not hasattr(original_tokenizer, "sp_model"):
            raise ValueError("The provided tokenizer does not have a SentencePiece model attached to it.")

        self.original_tokenizer = original_tokenizer
        self.model_string = original_tokenizer.sp_model.serialized_model_proto()

        m = ModelProto()
        m.ParseFromString(self.model_string)
        self.proto = m

        if self.proto.trainer_spec.byte_fallback:
            if not getattr(self, "handle_byte_fallback", None):
                logger.warning(
                    "The sentencepiece tokenization that you are converting to a fast tokenization uses the byte "
                    "fallback option which is not implemented in the fast tokenizers. In practice this means that the "
                    "fast version of the tokenization can produce unknown tokens whereas the sentencepiece version "
                    "would have converted these unknown tokens into a sequence of byte tokens matching the original "
                    "piece of text."
                )

    @staticmethod
    def vocab(proto):
        return [(piece.piece, piece.score) for piece in proto.pieces]

    @staticmethod
    def unk_id(proto):
        return proto.trainer_spec.unk_id

    def tokenizer(self, proto):
        model_type = proto.trainer_spec.model_type
        vocab_scores = self.vocab(proto)
        unk_id = self.unk_id(proto)

        if model_type == 1:
            tokenizer = Tokenizer(Unigram(vocab_scores, unk_id))
        elif model_type == 2:
            _, merges = SentencePieceExtractor(self.model_string).extract()
            bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
            tokenizer = Tokenizer(
                BPE(
                    bpe_vocab,
                    merges,
                    unk_token=proto.trainer_spec.unk_piece,
                    fuse_unk=True,
                )
            )
        else:
            raise Exception(
                "You're trying to run a `Unigram` model but you're file was trained with a different algorithm"
            )

        return tokenizer

    @staticmethod
    def normalizer(proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        _normalizers = [
            normalizers.Strip(left=False, right=True),  # stripping is important
            normalizers.Replace(Regex(" {2,}"), "▁"),
        ]
        if not precompiled_charsmap:
            return normalizers.Sequence(_normalizers)
        else:
            return normalizers.Sequence([normalizers.Precompiled(precompiled_charsmap)] + _normalizers)

    def pre_tokenizer(self, replacement, add_prefix_space):
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)

    @staticmethod
    def post_processor():
        return None

    def decoder(self, replacement, add_prefix_space):
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        return decoders.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        # control tokens are special user defined symbols are not both user and control tokens are AddedTokens Add
        # user defined symbols (type == 4) from sentencepiece (
        # https://github.com/google/sentencepiece/blob/6225e08edb2577757163b3f5dbba4c0b670ef445/src
        # /sentencepiece_model.proto#L299C29-L299C33)

        tokens_to_add = {
            id: AddedToken(token, normalized=False, special=special)
            for _id, token, special in [
                (_id, p.piece, p.type == 3) for _id, p in enumerate(self.proto.pieces) if p.type in [3, 4]
            ]
        }
        tokens_to_add = [k for _, k in sorted(tokens_to_add.items(), key=lambda x: x[0])]
        if len(tokens_to_add) > 0:
            # super hack: if a token.special is set, tokenization ignores it for now so FIXME @ArthurZ
            # Accumulate added tokens into batches of special/non-special tokens, because calling add_tokens() for
            # individual tokens would repeatedly rebuild a trie, which can be slow.
            is_last_special = None
            tokens = []
            for token in tokens_to_add:
                is_special = token.special
                if is_last_special is None or is_last_special == is_special:
                    tokens.append(token)
                else:
                    if is_last_special:
                        tokenizer.add_special_tokens(tokens)
                    else:
                        tokenizer.add_tokens(tokens)
                    tokens = [token]
                is_last_special = is_special
            if tokens:
                if is_last_special:
                    tokenizer.add_special_tokens(tokens)
                else:
                    tokenizer.add_tokens(tokens)
        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        return tokenizer
