import datasets
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default=None, type=str, required=True,
                    help="specify the language for which the text file needs to be created. Needs to be in ISO-2 format")
parser.add_argument("--partition", default=None, type=str, required=True,
                    help="Which data partition to use")

args = parser.parse_args()
lang = args.lang
partition = args.partition

train_text = []
dataset = datasets.load_dataset(f"WikiQuality/{partition}", data_dir=f"{lang}")['train']
for data in dataset:
    train_text.append(data['text'])

if not os.path.exists(f"./data/{lang}"):
    os.makedirs(f"./data/{lang}")

with open(f"./data/{lang}/{partition}.train.txt", "w") as f:
    f.write("\n".join(train_text))

valid_text = []
dataset = datasets.load_dataset(f"WikiQuality/{partition}", data_dir=f"{lang}")['test']
for data in dataset:
    valid_text.append(data['text'])

with open(f"./data/{lang}/{partition}.valid.txt", "w") as f:
    f.write("\n".join(valid_text))

