# File takes in a wikipedia dump in the format of links_lengths.csv, and splits it into 2 bins -
# high quality and low quality - based on the partition functions

import os

import nltk
import pandas as pd
import argparse
from collections import Counter
import transformers
import evaluate
import torch
import spacy
from tqdm import tqdm





def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", default='naija_pcm',
                        type=str, help="Location of folder containing the all_pages.csv file.")
    parser.add_argument("-p", "--partition", default='length',
                        type=str, help="Partition function to use. Options: length, links, red_pajamas")

    return parser.parse_args()

class Partition():
    def __init__(self, language):
        self.language = language
        self.df = pd.read_csv('wikis/' + self.language + '/all_pages.csv')
        self.articles = self.df['text'].tolist()

    def length(self):
        a_l = [(article, len(article)) for article in self.articles]
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
        unique_word_counts = []
        for article in self.df['text']:
            counts = Counter(article.split())
            unique_word_counts.append(len(counts))
        self.df['unique_word_counts'] = unique_word_counts
        mean = round(self.df['unique_word_counts'].mean())
        high_quality = self.df[self.df['unique_word_counts'] >= mean]
        low_quality = self.df[self.df['unique_word_counts'] < mean]
        print("Mean unique word count of articles: ", mean)
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        return high_quality, low_quality

    def links(self):
        mean = round(self.df['links'].mean())
        high_quality = self.df[self.df['links'] >= mean]
        low_quality = self.df[self.df['links'] < mean]
        print("Mean number of links of articles: ", mean)
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))

        return high_quality, low_quality

    # def perplexity(self):
    #     from spacy.lang.hi import Hindi
    #     nlp = Hindi()
    #     nlp.add_pipe('sentencizer')
    #     perplexity = evaluate.load("perplexity", module_type = 'metric')
    #     doc = nlp(self.articles[3])
    #     sents = list(doc.sents)
    #     predictions = []
    #     for sent in sents:
    #         predictions.append(sent.text.strip())
    #     print(predictions)
    #     results = perplexity.compute(
    #                                  model_id='xlm-roberta-base',
    #                                  # model_id='Davlan/afro-xlmr-base',
    #                                  add_start_token=False,
    #                                  predictions=predictions)
    #     print(list(results.keys()))
    #     print(results['mean_perplexity'])

    def perplexity(self):
        from spacy.lang.hi import Hindi
        nlp = Hindi()
        nlp.add_pipe('sentencizer')
        doc = nlp(self.articles[3])
        sents = list(doc.sents)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_id = 'xlm-roberta-base'
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        encodings = tokenizer('\n\n'.join([sent.text.strip() for sent in sents]), return_tensors='pt')

        max_length = model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        ppl = torch.exp(torch.stack(nlls).mean())
        print(ppl)




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
