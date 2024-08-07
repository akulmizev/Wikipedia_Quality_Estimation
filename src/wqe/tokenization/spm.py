import io
import logging
import os
import tempfile

from copy import deepcopy
from dataclasses import asdict
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from transformers import AddedToken, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import import_protobuf

from .base import BaseTokenizerMixin
from .utils import SpmConverter
from ..utils.config import TokenizerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPIECE_UNDERLINE = "▁"
VOCAB_FILES_NAMES = {"vocab_file": "tokenization.model", "tokenizer_file": "tokenization.json"}


class HfSentencePieceTokenizerBase(PreTrainedTokenizer, BaseTokenizerMixin):
    """
    HuggingFace Tokenizers interface for SentencePiece. Copied almost verbatim from:
    transformers.models.llama.tokenization_llama.py. The only changes are the removal of
    the chat templates and the addition of the `train_from_config` class method.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        sp_model_kwargs (`Dict[str, Any]`, `Optional`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to add spaces between special tokens.
        legacy (`bool`, *optional*):
            Whether or not the `legacy` behavior of the tokenization should be used. Legacy is before the merge
            of #24622 and #25224 which includes fixes to properly handle tokens that appear after special tokens.
            Make sure to also set `from_slow` to `True`.
        add_prefix_space (`bool`, *optional*, defaults to `True`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. Again, this should be set with `from_slow=True` to make sure it's taken into account.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            vocab_file: str,
            # unk_token: str = "<unk>",
            # bos_token: str = "<s>",
            # eos_token: str = "</s>",
            # pad_token: str = "<pad>",
            # sp_model: SentencePieceProcessor = None,
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            add_bos_token: bool = True,
            add_eos_token: bool = False,
            clean_up_tokenization_spaces: bool = False,
            spaces_between_special_tokens: bool = False,
            legacy: bool = None,
            add_prefix_space: bool = True,
            **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        # eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        # unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token
        # pad_token = AddedToken(pad_token, normalized=False, special=True) if isinstance(pad_token, str) else pad_token

        if legacy is None:
            legacy = True
            logger.warning(
                f"You are using the default legacy behaviour of the {self.__class__}. This is expected, and simply"
                " means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want"
                " to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thoroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenization from a"
                " GGUF file you can ignore this message"
            )

        self.legacy = legacy
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
        self.add_prefix_space = add_prefix_space

        super().__init__(
            # bos_token=bos_token,
            # eos_token=eos_token,
            # unk_token=unk_token,
            # pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            legacy=legacy,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    @property
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor
    def get_spm_processor(
            self,
            from_slow: bool = False
    ):
        tokenizer = SentencePieceProcessor(**self.sp_model_kwargs)
        if self.legacy or from_slow:  # no dependency on protobuf
            tokenizer.Load(self.vocab_file)
            return tokenizer

        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)

        return tokenizer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @classmethod
    def train_from_config(
            cls,
            dataset: List[str],
            config: TokenizerConfig,
            vocab_file: Optional[str] = None,
            **kwargs
    ):

        """
        Train a SentencePiece tokenization from a configuration.

        The config expects a model type to be specified, where only `bpe` and `unigram` are currently supported.
        If a normalizer is specified, it will be passed to SentencePiece as the `normalization_rule_name`,
        provided that the rule is supported by SentencePiece. `pre_tokenizer`, `decoder`, and `trainer` components
        will be ignored, as they are specific to HuggingFace tokenizers (theoretically I could write a full argument
        mapping, but I'm too lazy).

        Specific keyword arguments can also be called to the SentencePiece trainer by passing them to the
        config as `sp_kwargs`. See full set of options for `SentencePieceTrainer.train` here:

        https://github.com/google/sentencepiece/blob/master/doc/options.md

        Parameters
        ----------

        dataset : list of str
            A list of documents representing the dataset.
        config : TokenizerConfig
            A configuration object for the tokenization.
            See `wqe.utils.config.TokenizerConfig` for details.
        vocab_file : str, optional
            The path to save the vocabulary file. If `None`, the model will be saved to a temporary file.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        PreTrainedTokenizer
            An instance of the trained tokenization.
        """

        if any(["pre_tokenizer", "decoder", "trainer"]) in asdict(config).keys():
            logger.warning(
                "SentencePiece does not support pre_tokenizer, decoder, or trainer components. Ignoring."
            )

        if config.model.type not in ["unigram", "bpe"]:
            raise ValueError("Model type must be 'unigram' or 'bpe' for SentencePiece.")

        vocab_size = cls.predict_vocab_size(len("".join(dataset))) \
            if config.vocab_size == "auto" else config.vocab_size

        normalization_rule_name = config.normalizer.type if config.normalizer else "nmt_nfkc"
        sp_special_tokens = cls.convert_special_token_mapping(config.special_tokens)
        sp_kwargs = config.sp_kwargs if config.sp_kwargs else {}

        model = io.BytesIO()
        SentencePieceTrainer.train(
            sentence_iterator=iter(dataset),
            model_writer=model,
            model_type=config.model.type,
            normalization_rule_name=normalization_rule_name,
            vocab_size=vocab_size,
            **sp_special_tokens,
            **sp_kwargs
        )

        spm = SentencePieceProcessor(model_proto=model.getvalue())
        special_token_ids = {f"{k}_id": spm.piece_to_id(v) for k, v in config.special_tokens.items()}

        if vocab_file:
            os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
            writer = open(os.path.abspath(vocab_file), "wb")
            temp_file_path = vocab_file
        else:
            writer = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
            temp_file_path = writer.name
            logger.warning(f"No vocab file path provided. Saving to temporary file: {temp_file_path}.")

        writer.write(model.getvalue())
        writer.close()

        return cls(vocab_file=temp_file_path, **config.special_tokens, **special_token_ids, **kwargs)

    @staticmethod
    def convert_special_token_mapping(
            special_tokens_map: Dict[str, str]
    ) -> Dict[str, int]:

        if not all(token in special_tokens_map for token in ["pad_token", "unk_token"]):
            raise ValueError("Special tokens 'pad_token' and 'unk_token' must be provided.")

        map_copy = deepcopy(special_tokens_map)
        sp_map = {
            "pad_id": 0, "unk_id": 1,
            "bos_id": -1, "eos_id": -1
        }

        token_mapping = {
            "pad_token": "pad_piece",
            "unk_token": "unk_piece",
            "bos_token": "bos_piece",
            "eos_token": "eos_piece"
        }

        for old_key, new_key in token_mapping.items():
            if old_key in special_tokens_map:
                sp_map[new_key] = map_copy.pop(old_key)
                if old_key in ["bos_token", "eos_token"]:
                    sp_map[new_key.replace("piece", "id")] = 2 if old_key == "bos_token" else 3

        if map_copy:
            sp_map["user_defined_symbols"] = ",".join(map_copy.values())

        return sp_map

    @property
    def vocab_size(self):

        """Returns vocab size"""

        return self.sp_model.get_piece_size()

    def get_vocab(self):

        """Returns vocab as a dict"""

        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    def tokenize(
            self,
            text: str,
            **kwargs
    ) -> List[str]:

        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.
        """

        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)

        text = text.replace(SPIECE_UNDERLINE, " ")
        if self.add_prefix_space:
            text = SPIECE_UNDERLINE + text

        tokens = super().tokenize(text, **kwargs)

        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
    def _tokenize(self, text, **kwargs):

        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus, we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenization.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """

        tokens = self.sp_model.encode(text, out_type=str)
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        return tokens[self.unk_token_length:] if len(tokens) >= self.unk_token_length else tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):

        """Converts a sequence of tokens (string) in a single string."""

        # since we manually add the prefix space, we have to remove it when decoding
        if tokens[0].startswith(SPIECE_UNDERLINE) and self.add_prefix_space:
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0 and self.legacy:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                if prev_is_special and i == 1 and self.add_prefix_space and not token.startswith(SPIECE_UNDERLINE):
                    out_string += " "
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:

        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the vocabulary filename.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenization `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
                bos_token_id
                + ([0] * len(token_ids_0))
                + eos_token_id
                + bos_token_id
                + ([0] * len(token_ids_1))
                + eos_token_id
        )

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output


class HfSentencePieceTokenizer(PreTrainedTokenizerFast, BaseTokenizerMixin):
    """
    Hacky wrapper class to make use of the `PreTrainedTokenizerFast` interface with SentencePiece.
    This class will have to be extensively tested, as I'm not sure what backend integrations might be broken.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = HfSentencePieceTokenizerBase
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            tokenizer_object: HfSentencePieceTokenizerBase = None,
            vocab_file=None,
            **kwargs
    ):
        self.vocab_file = vocab_file

        if tokenizer_object:
            converter = SpmConverter(tokenizer_object)
            tokenizer_object = converter.converted()

        super().__init__(
            tokenizer_object=tokenizer_object,
            vocab_file=self.vocab_file,
            **kwargs
        )

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            **kwargs,
    ):

        tokenizer = HfSentencePieceTokenizerBase.from_pretrained(pretrained_model_name_or_path)

        return cls(tokenizer_object=tokenizer, vocab_file=tokenizer.vocab_file, **tokenizer.init_kwargs, **kwargs)

    @classmethod
    def train_from_config(
            cls,
            dataset: List[str],
            config: TokenizerConfig,
            vocab_file: Optional[str] = None,
            **kwargs
    ):

        slow_tokenizer = HfSentencePieceTokenizerBase.train_from_config(dataset, config, vocab_file, **kwargs)

        return cls(
            tokenizer_object=slow_tokenizer,
            vocab_file=slow_tokenizer.vocab_file,
            **slow_tokenizer.init_kwargs,
            **kwargs
        )

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
