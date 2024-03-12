import argparse
import json
import os

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, type=str, required=True,
                        help="Specify language for extracting Wiki dump. Needs to be in ISO-2 format \
                        (e.g. `en` for English)")
    parser.add_argument("--output_dir", default="./tokenizers", type=str, required=False,
                        help="Specify output directory where tokenizer vocabulary will be saved.")
    args = parser.parse_args()

    if not os.path.exists(f"./{args.output_dir}"):
        os.makedirs(f"./{args.output_dir}")

    vocab_mapper = json.load(open(f"./data/predicted_vocab.json", "r"))
    vocab_size = vocab_mapper[args.lang]

    dataset = load_dataset(f"WikiQuality/{args.lang}.filtered")['train']

    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.UnicodeScripts(),
         pre_tokenizers.Digits(),
         pre_tokenizers.Metaspace()]
        # pre_tokenizers.ByteLevel()]
    )
    tokenizer.decoder = decoders.Metaspace()
    # tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[SEP]", "[CLS]", "[UNK]"]
    )

    tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer, length=len(dataset))
    # tokenizer.save(f"./{args.output_dir}/wiki.{args.lang}.{len(tokenizer.get_vocab())}.json")
    tokenizer.save(f"./{args.output_dir}/wiki.{args.lang}.json")

if __name__ == "__main__":
    main()
