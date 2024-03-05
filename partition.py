
import os
import argparse
from collections import Counter
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tokenizers import Tokenizer
import numpy as np
import math
import re
from datasets import load_dataset
from datasets import enable_caching
import string
from nltk.util import ngrams
import pandas as pd
from scipy.stats.mstats import gmean

enable_caching()
def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", default=None, type=str, required=True,
                        help="Specify language for extracting Wiki dump. Needs to be in ISO-2 format \
                        (e.g. `pcm` for Naija)")
    parser.add_argument("-p", "--partition", default='length', required=True,
                        type=str, help="Partition function to use. Options: length, unique_words")
    parser.add_argument("--filtered", default=True, required=False,
                        help="Whether to use the filtered or raw version of the wikipedia")

    return parser.parse_args()

class Partition():
    def __init__(self, language, filtered):
        # self.language = language
        # self.df = pd.read_csv('wikis/' + self.language + '/all_pages.csv')
        # self.articles = self.df['text'].tolist()
        self.language = language
        # self.save_dir = 'wikis_cache/' + self.language + '/'
        if not os.path.exists('wikis_cache/'):
            os.mkdir('wikis_cache/')
        if filtered:
            print("Using filtered version of the wikipedia")
            self.dataset = load_dataset(f"WikiQuality/{self.language}_filtered",
                                                 cache_dir='wikis_cache/', split="train")
        else:
            self.dataset = load_dataset("wikimedia/wikipedia", f"20231101.{self.language}",
                                        streaming=False, cache_dir='wikis_cache/', split="train")

    def length(self):
        a_l = [(example['text'].strip(), len(example['text'].strip())) for example in self.dataset]
        cut_off = round(sum(l for _, l in a_l) / 2)
        a_l_sorted = sorted(a_l, key=lambda x: x[1], reverse=False)
        total_num_chars = 0
        high_quality = []
        low_quality = []
        for (article, length) in tqdm(a_l_sorted):
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


    def unique_words(self):
        unique_word_counts = {}  #unique subwords?- do that after tokenizers have been trained
        for example in tqdm(self.dataset):
            text = example['text']
            counts = Counter(text.split())
            unique_word_counts[example['id']] = len(counts)
        mean = int(np.mean(list(unique_word_counts.values())))
        # mean = gmean(list(unique_word_counts.values()))

        high_quality = [example['text'] for example in self.dataset if unique_word_counts[example['id']] >= mean]
        low_quality = [example['text'] for example in self.dataset if unique_word_counts[example['id']] < mean]

        print("Mean unique word count of articles: ", mean)
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        high_quality = '\n'.join(high_quality)
        low_quality = '\n'.join(low_quality)

        #Whats the difference between gmean and amean?

        # g_mean = gmean(list(unique_word_counts.values()))
        # a_mean = np.mean(list(unique_word_counts.values()))
        #
        # high_quality_amean = [(example['id'], example['text']) for example in self.dataset if unique_word_counts[example['id']] >= a_mean]
        # high_quality_gmean = [(example['id'], example['text']) for example in self.dataset if unique_word_counts[example['id']] >= g_mean]
        #
        # difference = [a for a in high_quality_gmean if a not in high_quality_amean]
        # print("Difference between high quality articles using gmean and amean: ", difference)

        #unique word count of the entire wiki
        # wiki = ' '.join([example['text'] for example in tqdm(self.dataset)])
        # unique_word_counts = Counter(wiki.split())
        # word_counts = len(wiki.split())
        # char_counts = len(wiki)
        # print("Unique word count of the entire wiki: ", len(unique_word_counts))
        # print("Total word count of the entire wiki: ", word_counts)
        # print("Total character count of the entire wiki: ", char_counts)

        # return high_quality, low_quality

    def unique_subwords(self):
        tokeniser = Tokenizer.from_file(f'tokenizers/wiki.{self.language}.json')
        unique_subword_counts = {}
        for example in tqdm(self.dataset):
            text = example['text'].strip()
            tokens = tokeniser.encode(text).tokens
            counts = Counter(tokens)
            words = text.split()
            # unique_subword_counts[example['text']] = len(counts)
            # use this to calculate fertility
            unique_subword_counts[example['text']] = len(counts), len(tokens), len(words)

        # mean = int(np.mean(list(unique_subword_counts.values())))
        # # mean = gmean([n+1 for n in unique_subword_counts.values()])
        # high_quality = [k for k,v in tqdm(unique_subword_counts.items()) if v >= mean]
        # low_quality = [k for k,v in tqdm(unique_subword_counts.items()) if v < mean]
        #
        #
        # print("Mean unique subword count of articles: ", mean)
        # print("Number of articles in high quality bin: ", len(high_quality))
        # print("Number of articles in low quality bin: ", len(low_quality))
        # high_quality = '\n'.join(high_quality)
        # low_quality = '\n'.join(low_quality)
        #
        # return high_quality, low_quality

        # calculate fertility - comment out the above and uncomment this
        fertility= sum([v[1] for v in unique_subword_counts.values()])/sum([v[2] for v in unique_subword_counts.values()])
        print("Fertility of the language: ", fertility)




    def n_grams(self):
        print("Filtering on unique trigram count...")
        # total_bigrams = {}
        total_trigrams = {}
        for example in tqdm(self.dataset):
            text = example['text'].strip()
            words = text.split()
            # bigrams = list(ngrams(words, 2))
            trigrams = list(ngrams(words, 3))
            # count_bigrams = Counter(bigrams)
            count_trigrams = Counter(trigrams)
            # total_bigrams[example['text']] = len(count_bigrams)
            total_trigrams[example['text']] = len(count_trigrams)

        # mean_bigrams = np.mean(list(total_bigrams.values()))

        mean_trigrams = np.mean(list(total_trigrams.values()))
        # mean_trigrams = gmean([n+1 for n in total_trigrams.values()])

        #replace with mean_bigrams and uncomment the bigram stuff above to filter on bigrams
        high_quality = [k for k,v in tqdm(total_trigrams.items()) if v >= mean_trigrams]
        low_quality = [k for k,v in tqdm(total_trigrams.items()) if v < mean_trigrams]

        print("Mean unique trigram count of articles: ", mean_trigrams)
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        high_quality = '\n'.join(high_quality)
        low_quality = '\n'.join(low_quality)

        return high_quality, low_quality

    def word_length(self):
        print("Filtering mean word length...")
        mean_word_length = {}
        overall_mean_word_length = []
        for example in tqdm(self.dataset):
            text = example['text'].strip()
            words = text.split()
            len_words = [len(word) for word in words]
            for length in len_words:
                overall_mean_word_length.append(length)
            mean_len = np.mean(len_words)
            mean_word_length[example['id']] = [mean_len, example['text']]
        overall_mean = np.mean(overall_mean_word_length)
        # overall_mean = gmean(overall_mean_word_length)
        print("Overall mean word length: " + str(overall_mean))
        high_quality = [example[1] for example in tqdm(mean_word_length.values()) if example[0] >= overall_mean]
        low_quality = [example[1] for example in tqdm(mean_word_length.values()) if example[0] < overall_mean]

        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        high_quality = '\n'.join(high_quality)
        low_quality = '\n'.join(low_quality)

        return high_quality, low_quality
        # return high_quality

    def english_chars(self):
        english_re = re.compile(r'[A-Za-z]')  #get ratio between ascii and non-ascii characters
        num_english_chars = 0

        high_quality = []
        low_quality = []
        # for example in tqdm(self.dataset):   #splitting on ratio of english characters
        #     text = example['text'].strip()
        #     match = 0
        #     for char in text:
        #         if english_re.match(char):
        #             match += 1
        #     total = len(text)
        #     ratio = match/total
        #     if ratio > 0.2:
        #         low_quality.append(text)
        #     else:
        #         high_quality.append(text)
        eng_chars_match = {}
        for example in tqdm(self.dataset):   #splitting on average
            text = example['text'].strip()
            match = 0
            for char in text:
                if english_re.match(char):
                    match += 1
            eng_chars_match[example['text']] = match

        mean_eng_chars = np.mean(list(eng_chars_match.values()))
        # mean_eng_chars = gmean([n+1 for n in eng_chars_match.values()])
        high_quality = [k for k,v in tqdm(eng_chars_match.items()) if v <= mean_eng_chars]
        low_quality = [k for k,v in tqdm(eng_chars_match.items()) if v > mean_eng_chars]

        print("Mean number of english characters: ", mean_eng_chars)
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        high_quality = '\n'.join(high_quality)
        low_quality = '\n'.join(low_quality)

        return high_quality, low_quality

    def stupid_filters(self):

        article_length = []
        unique_word_counts = {}
        unique_subword_counts = {}
        total_trigrams = {}
        mean_word_length = {}
        overall_mean_word_length = []
        english_re = re.compile(r'[A-Za-z]')
        tokeniser = Tokenizer.from_file(f'tokenizers/wiki.{self.language}.json')
        # low_quality_english_chars = []
        eng_chars_match = {}
        for example in tqdm(self.dataset):
            text = example['text'].strip()

            article_length.append((example['text'], len(text)))

            words = text.split()
            word_counts = Counter(words)
            tokens = tokeniser.encode(text).tokens
            token_counts = Counter(tokens)
            unique_word_counts[example['text']] = len(word_counts)
            unique_subword_counts[example['text']] = len(token_counts)

            trigrams = list(ngrams(words, 3))
            count_trigrams = Counter(trigrams)
            total_trigrams[example['text']] = len(count_trigrams)

            word_lengths = [len(word) for word in words]
            for length in word_lengths:
                overall_mean_word_length.append(length)
            mean_len_article = np.mean(word_lengths)
            mean_word_length[example['text']] = mean_len_article

            if self.language != 'pcm':
                match = 0
                total = len(text)
                for char in text:
                    if english_re.match(char):
                        match += 1
                eng_chars_match[example['text']] = match/total



        # print("Number of articles in low quality english chars: ", len(low_quality_english_chars))

        print("On length...")
        article_length_sorted = sorted(article_length, key=lambda x: x[1], reverse=False)
        length_cutoff = round(sum(l for _, l in article_length)/2)
        total_num_chars = 0
        low_quality_length = []
        for (article, length) in tqdm(article_length_sorted):
            if total_num_chars < length_cutoff:
                low_quality_length.append(article.strip())
                total_num_chars += length
            else:
                total_num_chars += length
        print("Number of articles in low quality length: ", len(low_quality_length))

        print("On unique words...")
        unique_word_cutoff = int(np.mean(list(unique_word_counts.values())))
        low_quality_unique_words = [key for key, value in unique_word_counts.items() if value < unique_word_cutoff]
        print("Number of articles in low quality unique words: ", len(low_quality_unique_words))

        print("On unique subwords...")
        mean_subword_count = int(np.mean(list(unique_subword_counts.values())))
        low_quality_subwords = [key for key, value in unique_subword_counts.items() if value < mean_subword_count]
        print("Number of articles in low quality subwords: ", len(low_quality_subwords))

        print("On trigrams...")

        mean_trigram_count = int(np.mean(list(total_trigrams.values())))
        low_quality_ngrams = [key for key, value in total_trigrams.items() if value < mean_trigram_count]
        print("Number of articles in low quality ngrams: ", len(low_quality_ngrams))

        print("On word length...")

        mean_word_len_cutoff = np.mean(overall_mean_word_length)
        low_quality_wordlength = [key for key, value in mean_word_length.items() if value < mean_word_len_cutoff]
        print("Number of articles in low quality word length: ", len(low_quality_wordlength))

        print("On english chars...")
        mean_eng_chars = np.mean(list(eng_chars_match.values()))
        low_quality_english_chars = [k for k,v in tqdm(eng_chars_match.items()) if v > mean_eng_chars]
        print("Number of articles in low quality english chars: ", len(low_quality_english_chars))

        low_quality = []
        high_quality = []
        for example in tqdm(self.dataset):
            if example['text'] in low_quality_wordlength or \
                example['text'] in low_quality_ngrams or \
                example['text'] in low_quality_unique_words or \
                example['text'] in low_quality_subwords or \
                example['text'] in low_quality_length or \
                example['text'] in low_quality_english_chars:
                low_quality.append(example['text'])
            else:
                high_quality.append(example['text'])

        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        high_quality = '\n'.join(high_quality)
        low_quality = '\n'.join(low_quality)

        return high_quality, low_quality

    def stats(self):
        tokenizer = Tokenizer.from_file(f'tokenizers/wiki.{self.language}.json')
        article_length = {}
        unique_word_counts = {}
        unique_subword_counts = {}
        total_trigrams = {}
        word_length = {}
        overall_mean_word_length = []
        english_re = re.compile(r'[A-Za-z]')
        english_chars = {}
        for example in tqdm(self.dataset):
            text = example['text'].strip()
            words = text.split()

            article_length[example['id']] = len(text)

            counts = Counter(words)
            unique_word_counts[example['id']] = len(counts)

            tokens = tokenizer.encode(text).tokens
            token_counts = Counter(tokens)
            unique_subword_counts[example['id']] = len(token_counts)

            trigrams = list(ngrams(words, 3))
            count_trigrams = Counter(trigrams)
            total_trigrams[example['id']] = len(count_trigrams)

            word_lengths = [len(word) for word in words]
            for length in word_lengths:
                overall_mean_word_length.append(length)
            mean_len_article = np.mean(word_lengths)
            word_length[example['id']] = mean_len_article

            if self.language != 'pcm':
                match = 0 #get ratio with not-match chars also
                total = len(text)
                for char in text:
                    if english_re.match(char):
                        match += 1
                english_chars[example['id']] = match/total

        df_article_length = pd.DataFrame(article_length.items(), columns=['id', 'length'])
        df_unique_word_counts = pd.DataFrame(unique_word_counts.items(), columns=['id', 'unique_words'])
        df_unique_subword_counts = pd.DataFrame(unique_subword_counts.items(), columns=['id', 'unique_subwords'])
        df_total_trigrams = pd.DataFrame(total_trigrams.items(), columns=['id', 'trigrams'])
        df_mean_word_length = pd.DataFrame(word_length.items(), columns=['id', 'mean_word_length'])
        df_english_chars = pd.DataFrame(english_chars.items(), columns=['id', 'english_chars'])
        stats_dir = 'stats_' + self.language + '/'
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

        df_article_length.to_csv(stats_dir + 'article_length.csv', index=False)
        df_unique_word_counts.to_csv(stats_dir + 'unique_word_counts.csv', index=False)
        df_unique_subword_counts.to_csv(stats_dir + 'unique_subword_counts.csv', index=False)
        df_total_trigrams.to_csv(stats_dir + 'total_trigrams.csv', index=False)
        df_mean_word_length.to_csv(stats_dir + 'mean_word_length.csv', index=False)
        if self.language != 'pcm':
            df_english_chars.to_csv(stats_dir + 'english_chars.csv', index=False)













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
            overall_perplexity[example['id']] = np.mean(ex_perp)



        # for chunk level perplexity
        # for example in tqdm(self.dataset):
        #     text = example['text'].split('\n')
        #     ex_perp = []
        #     for sent in text:
        #         try:
        #             if sent == '' or sent == ' ':
        #                 continue
        #             sent = sent.strip()
        #             tokenize_input = tokenizer.tokenize(sent, truncation=True, max_length=512)
        #             tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).cuda()
        #             with torch.no_grad():
        #                 loss = model(tensor_input, labels=tensor_input)[0]
        #             result = np.exp(loss.cpu().detach().numpy()) ##change this to make it faster
        #             ex_perp.append(result)
        #         except:
        #             print("this is the sentence: ", sent)
        #             continue
        #     overall_perplexity[example['id']] = sum(ex_perp) / len(ex_perp)


        # for article level perplexity

        # for example in tqdm(self.dataset):
        #     tokenize_input = tokenizer.tokenize(example['text'], truncation=True, max_length=512)
        #     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).cuda()
        #     with torch.no_grad():
        #         loss = model(tensor_input, labels=tensor_input)[0]
        #     result = np.exp(loss.cpu().detach().numpy())
        #     overall_perplexity[example['id']] = result


        # mean = round(sum(list(overall_perplexity.values())) / len(overall_perplexity))
        median = np.median(list(overall_perplexity.values()))
        high_quality = [example['text'] for example in self.dataset if overall_perplexity[example['id']] >= median]
        low_quality = [example['text'] for example in self.dataset if overall_perplexity[example['id']] < median]
        print("Median perplexity: ", median)
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))

        high_quality = '\n'.join(high_quality)
        low_quality = '\n'.join(low_quality)
        return high_quality, low_quality





def main():
    args = create_arg_parser()
    if args.partition == 'length':
        bins = Partition(args.language, args.filtered).length()  # high quality, low quality
    if args.partition == 'unique_words':
        bins = Partition(args.language, args.filtered).unique_words()
    if args.partition == 'unique_subwords':
        bins = Partition(args.language, args.filtered).unique_subwords()
    if args.partition == 'perplexity':
        bins = Partition(args.language, args.filtered).perplexity()
    if args.partition == 'n_grams':
        bins = Partition(args.language, args.filtered).n_grams()
    if args.partition == 'word_length':
        bins = Partition(args.language, args.filtered).word_length()
    if args.partition == 'english_chars':
        bins = Partition(args.language, args.filtered).english_chars()
    if args.partition == 'stupid_filters':
        bins = Partition(args.language, args.filtered).stupid_filters()
    if args.partition == 'stats':
        Partition(args.language, args.filtered).stats()



        with open('wikis/' + args.language + '/' + args.partition + 'high_quality.txt', 'w+') as high_quality:
            high_quality.write(bins[0])
        with open('wikis/' + args.language + '/' + args.partition + 'low_quality.txt', 'w+') as low_quality:
            low_quality.write(bins[1])

if __name__ == '__main__':
    main()
