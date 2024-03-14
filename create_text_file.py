import datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default=None, type=str, required=True,
                    help="specify the language for which the text file needs to be created. Needs to be in ISO-2 format")

args = parser.parse_args()
lang = args.lang

text = []
dataset = datasets.load_dataset(f"WikiQuality/{lang}_filtered")['train']
for data in dataset:
    text.append(data['text'])

with open(f"./data/{lang}/wiki_{lang}.txt", "w") as f:
    f.write("\n".join(text))
