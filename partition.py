# File takes in a wikipedia dump in the format of links_lengths.csv, and splits it into 2 bins -
# high quality and low quality - based on the partition functions

import os

import nltk
import pandas as pd
import argparse
from collections import Counter
# import transformers
import evaluate
import torch
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import math
import re
import datasets



def create_arg_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-l", "--language", default='naija_pcm',
    #                     type=str, help="Location of folder containing the all_pages.csv file.")
    parser.add_argument("-l", "--language", default=None, type=str, required=True,
                        help="Specify language for extracting Wiki dump. Needs to be in ISO-2 format \
                        (e.g. `pcm` for Naija)")
    parser.add_argument("-p", "--partition", default='length',
                        type=str, help="Partition function to use. Options: length, links, red_pajamas")

    return parser.parse_args()

class Partition():
    def __init__(self, language):
        # self.language = language
        # self.df = pd.read_csv('wikis/' + self.language + '/all_pages.csv')
        # self.articles = self.df['text'].tolist()
        self.language = language
        # self.save_dir = 'wikis_cache/' + self.language + '/'
        # if not os.path.exists(self.save_dir):
        #     os.mkdir(self.save_dir)
        self.dataset = datasets.load_dataset("wikimedia/wikipedia", f"20231101.{self.language}",
                                        streaming=False, cache_dir='wikis_cache/', split="train")

    def length(self):
        a_l = [(example['text'].strip(), len(example['text'].strip())) for example in self.dataset]
        cut_off = round(sum(l for _, l in a_l) / 2)
        a_l_sorted = sorted(a_l, key=lambda x: x[1], reverse=False)
        total_num_chars = 0
        high_quality = []
        low_quality = []
        for (article, length) in a_l_sorted:
            if total_num_chars < cut_off:
                low_quality.append(article.strip())
                total_num_chars += length
            else:
                high_quality.append(article.strip())
                total_num_chars += length
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        print("Total number of characters in high quality bin: ", sum(len(article) for article in high_quality))
        print("Total number of characters in low quality bin: ", sum(len(article) for article in low_quality))
        high_quality = '\n'.join(high_quality)
        low_quality = '\n'.join(low_quality)

        return high_quality, low_quality

    def red_pajamas(self):
        unique_word_counts = {}
        for example in self.dataset:
            text = example['text']
            counts = Counter(text.split())
            unique_word_counts[example['id']] = len(counts)
        mean = round(sum(list(unique_word_counts.values())) / len(unique_word_counts))
        high_quality = [example['text'] for example in self.dataset if unique_word_counts[example['id']] >= mean]
        low_quality = [example['text'] for example in self.dataset if unique_word_counts[example['id']] < mean]
        print("Mean unique word count of articles: ", mean)
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        return high_quality, low_quality


    def perplexity(self):
        tokenizer = AutoTokenizer.from_pretrained('Davlan/afro-xlmr-base')
        model = AutoModelForMaskedLM.from_pretrained('Davlan/afro-xlmr-base')
        model.eval()
        model.to('cuda')

        overall_perplexity = {}

        ## for sentence level perplexity
        from spacy.lang.en import English
        nlp = English()
        nlp.add_pipe("sentencizer")

        for example in tqdm(self.dataset):
            text = example['text'].strip()
            doc = nlp(text)
            sentences = list(doc.sents)
            ex_perp = []
            for sent in sentences:
                sent = str(sent).strip()
                tokenize_input = tokenizer.tokenize(sent, truncation=True, max_length=512)
                tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).cuda()
                with torch.no_grad():
                    loss = model(tensor_input, labels=tensor_input)[0]
                result = np.exp(loss.cpu().detach().numpy())
                ex_perp.append(result)
            overall_perplexity[example['id']] = sum(ex_perp) / len(ex_perp)


        ## for article level perplexity
        #
        # for example in tqdm(self.dataset):
        #     tokenize_input = tokenizer.tokenize(example['text'], truncation=True, max_length=512)
        #     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).cuda()
        #     with torch.no_grad():
        #         loss = model(tensor_input, labels=tensor_input)[0]
        #     result = np.exp(loss.cpu().detach().numpy())
        #     overall_perplexity[example['id']] = result


        mean = round(sum(list(overall_perplexity.values())) / len(overall_perplexity))
        high_quality = [example['text'] for example in self.dataset if overall_perplexity[example['id']] >= mean]
        low_quality = [example['text'] for example in self.dataset if overall_perplexity[example['id']] < mean]
        print("Mean perplexity: ", mean)
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        return high_quality, low_quality





def main():
    args = create_arg_parser()
    if args.partition == 'length':
        bins = Partition(args.language).length()
    if args.partition == 'red_pajamas':
        bins = Partition(args.language).red_pajamas()
    if args.partition == 'links':
        bins = Partition(args.language).links()
    if args.partition == 'perplexity':
        Partition(args.language).perplexity()


        # with open('wikis/' + args.language + '/' + args.partition + 'high_quality.txt', 'w+') as high_quality:
        #     high_quality.write(bins[0])
        # with open('wikis/' + args.language + '/' + args.partition + 'low_quality.txt', 'w+') as low_quality:
        #     low_quality.write(bins[1])

if __name__ == '__main__':
    main()
