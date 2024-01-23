
import os
import argparse
from collections import Counter
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import math
import re
import datasets
import string


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", default=None, type=str, required=True,
                        help="Specify language for extracting Wiki dump. Needs to be in ISO-2 format \
                        (e.g. `pcm` for Naija)")
    parser.add_argument("-p", "--partition", default='length', required=True,
                        type=str, help="Partition function to use. Options: length, links, unique_words")

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


    def unique_words(self):
        unique_word_counts = {}
        for example in self.dataset:
            text = example['text']
            counts = Counter(text.split())
            unique_word_counts[example['id']] = len(counts)
        mean = int(np.mean(list(unique_word_counts.values())))
        high_quality = [example['text'] for example in self.dataset if unique_word_counts[example['id']] >= mean]
        low_quality = [example['text'] for example in self.dataset if unique_word_counts[example['id']] < mean]
        print("Mean unique word count of articles: ", mean)
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        high_quality = '\n'.join(high_quality)
        low_quality = '\n'.join(low_quality)

        return high_quality, low_quality

    def n_grams(self):
        from nltk.util import ngrams
        total_bigrams = {}
        total_trigrams = {}
        for example in self.dataset:
            text = example['text'].strip()
            words = text.split()
            bigrams = list(ngrams(words, 2))
            trigrams = list(ngrams(words, 3))
            count_bigrams = Counter(bigrams)
            count_trigrams = Counter(trigrams)
            total_bigrams[example['id']] = len(count_bigrams)
            total_trigrams[example['id']] = len(count_trigrams)

        mean_bigrams = np.mean(list(total_bigrams.values()))
        mean_trigrams = np.mean(list(total_trigrams.values()))

        #just replace trigrams with bigrams to partition based on unique bigrams

        high_quality = [example['text'] for example in self.dataset if total_trigrams[example['id']] >= mean_trigrams]
        low_quality = [example['text'] for example in self.dataset if total_trigrams[example['id']] < mean_trigrams]
        print("Mean unique trigram count of articles: ", mean_trigrams)
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        high_quality = '\n'.join(high_quality)
        low_quality = '\n'.join(low_quality)

        return high_quality, low_quality




    def word_length(self):
        mean_word_length = {}
        overall_mean_word_length = []
        for example in self.dataset:
            text = example['text'].strip()
            words = text.split()
            len_words = [len(word) for word in words]
            for length in len_words:
                overall_mean_word_length.append(length)
            mean_len = np.mean(len_words)
            mean_word_length[example['id']] = mean_len
        overall_mean = np.mean(overall_mean_word_length)
        print("Overall mean word length: " + str(overall_mean))
        high_quality = [example['text'] for example in self.dataset if mean_word_length[example['id']] >= overall_mean]
        low_quality = [example['text'] for example in self.dataset if mean_word_length[example['id']] < overall_mean]
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        high_quality = '\n'.join(high_quality)
        low_quality = '\n'.join(low_quality)

        return high_quality, low_quality

    def english_words(self):
        english_re = re.compile(r'[A-Za-z]')
        num_english_chars = 0
        num_punct_chars = 0
        total_char_count = 0
        eng_to_lang_ratio = {}
        for example in self.dataset:
            num_example_english_chars = 0
            num_example_punct_chars = 0
            example_char_count = 0
            text = example['text'].strip()
            words = text.split()
            for word in words:
                for char in word:
                    total_char_count += 1
                    example_char_count += 1
                    if english_re.match(char):
                        num_example_english_chars += 1
                        num_english_chars +=1
                    elif char in string.punctuation:
                        num_example_punct_chars += 1
                        num_punct_chars += 1
            eng_to_lang_ratio[example['id']] = num_example_english_chars/example_char_count
        total_ratio = num_english_chars/total_char_count
        print(f"Total English to {self.language} ratio is {total_ratio}")

        high_quality = [example['text'] for example in self.dataset if eng_to_lang_ratio[example['id']] <= 0]
        low_quality = [example['text'] for example in self.dataset if eng_to_lang_ratio[example['id']] > 0]
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        high_quality = '\n'.join(high_quality)
        low_quality = '\n'.join(low_quality)

        return high_quality, low_quality

    def stupid_filters(self):
        print("it's working")
        english_re = re.compile(r'[A-Za-z]')

        print("""#########################################################\n
        #### Please ignore the print statements here until prompted otherwise #####\n
        #####################################################""")
        _, low_quality_wordlength = self.word_length()
        _, low_quality_ngrams = self.n_grams()
        _, low_quality_unique_words = self.unique_words()
        _, low_quality_length = self.length()
        low_quality = []
        high_quality = []
        for example in self.dataset:
            text = example['text']
            if self.language != 'pcm' and english_re.match(text):
                low_quality.append(text)
                continue
            elif text in low_quality_ngrams:
                low_quality.append(text)
                continue
            elif text in low_quality_wordlength:
                low_quality.append(text)
                continue
            elif text in low_quality_unique_words:
                low_quality.append(text)
                continue
            elif text in low_quality_length:
                low_quality.append(text)
                continue
            else:
                high_quality.append(text)

        print("""#########################################################\n
            #### This is for the splits generated in this partition function #####\n
            #####################################################""")
        print("Number of articles in high quality bin: ", len(high_quality))
        print("Number of articles in low quality bin: ", len(low_quality))
        print(high_quality[0])
        high_quality = '\n'.join(high_quality)
        low_quality = '\n'.join(low_quality)
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
        bins = Partition(args.language).length()  # high quality, low quality
    if args.partition == 'unique_words':
        bins = Partition(args.language).unique_words()
    if args.partition == 'perplexity':
        bins = Partition(args.language).perplexity()
    if args.partition == 'n_grams':
        bins = Partition(args.language).n_grams()
    if args.partition == 'word_length':
        bins = Partition(args.language).word_length()
    if args.partition == 'english_words':
        bins = Partition(args.language).english_words()
    if args.partition == 'stupid_filters':
        Partition(args.language).stupid_filters()



        # with open('wikis/' + args.language + '/' + args.partition + 'high_quality.txt', 'w+') as high_quality:
        #     high_quality.write(bins[0])
        # with open('wikis/' + args.language + '/' + args.partition + 'low_quality.txt', 'w+') as low_quality:
        #     low_quality.write(bins[1])

if __name__ == '__main__':
    main()
