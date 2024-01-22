import argparse
import os

import datasets
import pandas as pd

from tqdm import tqdm


def get_stats_dict(article):
    article_stats = {'id': article['id'],
                     'char_count': len(article['text']),
                     'byte_count': len(article['text'].encode('utf-8'))}

    return article_stats


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--lang", default=None, type=str, required=True,
                        help="Specify language for extracting Wiki dump. Needs to be in ISO-2 format \
                        (e.g. `en` for English)")
    parser.add_argument("--dump_date", default="20231101", type=str, required=False,
                        help="Specify date for dump that needs to be extracted (e.g. `20231101`). \
                        Dump dates can be found at https://dumps.wikimedia.org/{iso-2}wiki/.")
    parser.add_argument("--save_dir", default=None, type=str, required=True,
                        help="Specify directory where datasets should be save. ")
    parser.add_argument("--stats_dir", default=None, type=str, required=True,
                        help="Specify directory where dataset statistics (char and byte count per doc) \
                        will be saved. Will skip collection if left unspecified.")
    parser.add_argument("--streaming", default=True, type=bool, required=False,
                        help="Specify if dataset should be streamed and not downloaded in full. \
                        Useful for extracting stats or filtering data.")

    args = parser.parse_args()

    try:
        dataset = datasets.load_dataset("wikimedia/wikipedia", f"{args.dump_date}.{args.lang}",
                                        streaming=args.streaming, cache_dir=args.save_dir)

        if args.stats_dir:
            if not os.path.exists(args.stats_dir):
                os.mkdir(args.stats_dir)

            stats = map(get_stats_dict, tqdm(dataset['train'], desc="Collecting stats..."))
            out = pd.DataFrame(stats)
            out.to_csv(f"{args.stats_dir}/{args.lang}.char_and_byte_count.per_doc.csv", index=False)
        else:
            print(f"Skipping stat collection.")
            exit()

    except FileNotFoundError:
        print(f"Could not find dump for {args.lang} on date: {args.dump_date}. Skipping!")
        exit()


if __name__ == "__main__":
    main()
