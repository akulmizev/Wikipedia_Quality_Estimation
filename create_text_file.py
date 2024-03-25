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

text = []
dataset = datasets.load_dataset(f"WikiQuality/{lang}.{partition}")['train']
for data in dataset:
    text.append(data['text'])

if not os.path.exists(f"./data/{lang}"):
    os.makedirs(f"./data/{lang}")

with open(f"./data/{lang}/wiki_{lang}.txt", "w") as f:
    f.write("\n".join(text))
