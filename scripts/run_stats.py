# from wqe import WikiLoader
import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from datasets import load_dataset
from nltk.util import ngrams
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.special import softmax
from sklearn.feature_selection import VarianceThreshold
from transformers import PreTrainedTokenizerFast
from kneed import KneeLocator
from wqe.data.metrics import LengthCharacters, LengthWords, UniqueWords, UniqueTrigrams, UniqueCharacters, AlphaChars, \
    NumLines, FracAllCapsWords, FracNoScriptWords, MeanWordLength, FracUniqueWords, FracUniqueTrigrams, \
    FracUniqueCharacters, UnigramEntropy, TrigramEntropy, CharacterEntropy, LinesEndWithPunctuation, NumWordsPerLine, \
    Perplexity

# best langs 'ig',

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, default='ceb')
parser.add_argument('--partition', type=str, default='pre_filtered')
parser.add_argument('--metric', type=str, default='unigram_entropy')
parser.add_argument('--quality', type=str, default='all')
parser.add_argument('--overlap', type=bool, default=False)
parser.add_argument('--bins', type=int, default=1000)
parser.add_argument('--sample', type=bool, default=False)
parser.add_argument('--xlim', type=int, default=None) #set to 20k for length, 2k for rest
args = parser.parse_args()

tokenizer = PreTrainedTokenizerFast.from_pretrained(f"WikiQuality/pre_filtered.{args.lang}")
function_map = {
    "length_chars": LengthCharacters,
    # "length_words": LengthWords,
    "unique_words": UniqueWords,
    "unique_trigrams": UniqueTrigrams,
    # "unique_chars": UniqueCharacters,
    # "alpha_chars": AlphaChars,
    # "num_lines": NumLines,
    # "frac_all_caps_words": FracAllCapsWords,
    # "frac_no_script_words": FracNoScriptWords,
    # "doc_mean_word_length": MeanWordLength,
    "frac_unique_words": FracUniqueWords,
    "frac_unique_trigrams": FracUniqueTrigrams,
    # "frac_unique_chars": FracUniqueCharacters,
    "unigram_entropy": UnigramEntropy,
    "trigram_entropy": TrigramEntropy,
    # "char_entropy": CharacterEntropy,
    # "lines_end_with_punctuation": LinesEndWithPunctuation,
    # "num_words_per_line": NumWordsPerLine,
    # "perplexity": Perplexity
}


if args.quality == 'all':
    data = load_dataset(f"WikiQuality/pre_filtered", args.lang)
    data = datasets.concatenate_datasets([data['train'], data['test']])
    data = data.map(lambda x:{'tokens' : tokenizer.tokenize(x['text'])})
    if args.metric == 'all':
        for metric in function_map.keys():
            data = data.map(lambda x: {metric: function_map[metric].calculate(x)[metric]})
        df = pd.DataFrame(data, columns=data.features.keys())
        for metric in function_map.keys():
            df[metric] = df[metric].astype(float)
            arr = np.array(df[metric].tolist())
            arr_norm = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
            df[metric] = list(arr_norm)
    else:
        data = data.map(lambda x: {args.metric: function_map[args.metric].calculate(x)[args.metric]})
        df = pd.DataFrame(data, columns=data.features.keys())
        df[args.metric] = df[args.metric].astype(float)
        for index, row in df.iterrows():
            if int(row[args.metric]) == 8:
                print(row['url'])

else:
    # data_partitioned = load_dataset(f"WikiQuality/{args.partition}_{args.quality}", args.lang)
    data_partitioned = load_dataset(f"WikiQuality/pre_filtered", args.lang)
    data_partitioned = datasets.concatenate_datasets(data_partitioned['train'], data_partitioned['test'])
    data_partitioned = data_partitioned.map(lambda x: {args.metric: function_map[args.metric](x)})
    df = pd.DataFrame(data_partitioned, columns=data_partitioned.features.keys())
    df = df.sample(2504)
    df[args.metric] = df[args.metric].astype(float)
    arr = np.array(df[args.metric].tolist())
    arr_norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    df[args.metric] = list(arr_norm)
    if args.overlap:
        data_all = load_dataset("WikiQuality/pre_filtered", args.lang)
        data_all = data_all.map(lambda x: {args.metric: function_map[args.metric](x['text'])})
        df_all = pd.DataFrame(data_all['train'], columns=data_all['train'].features.keys())
        df_all[args.metric] = df_all[args.metric].astype(float)
        arr = np.array(df_all[args.metric].tolist())
        arr_norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 100
        df_all[args.metric] = list(arr_norm)
        if args.sample:
            df_all = df_all.sort_values(by=args.metric, ascending=True)
            df_all = df_all[:2504]


if args.metric == 'all':
    df['All'] = df['char_entropy'] + df['unigram_entropy'] + df['trigram_entropy']
    sorted = df['All'].sort_values().tolist()
else:
    sorted = df[args.metric].sort_values().tolist()

# best = sorted[:2504]
# random_sample = np.random.choice(np.array(sorted), size=1000, replace=False)
# new = [i for i in sorted if i < 25]
# print(len(new))
# fig, ax = plt.subplots()
# sns.histplot([best, random_sample], kde=True, legend=True)
# plt.title('al_metrics')
# plt.legend(['Random', 'Worst'])
# plt.show()

# variances = [np.var(sorted[:i]) for i in range(1, len(sorted))]
# variances = variances[np.argmax(variances):]
# std = np.std(variances)
# variances = np.array(variances)
# variances = variances[abs(variances - np.mean(variances)) < 2 * np.std(variances)]
# n_values = np.arange(0, len(sorted))
# knee = KneeLocator(n_values, sorted, curve='convex', direction='increasing', interp_method='polynomial')
# #
# knee.plot_knee(title=args.lang)
# plt.title(f'{args.lang} - {args.partition}_{args.quality}')
# plt.show()
# elbow_n = knee.all_knees
# print(f'The elbow point is at n = {elbow_n}')
# print(f'The elbow point is at value = {len(sorted) - knee.knee}')
# std = []
# for val in range(len(sorted)):
#     if val != 0:
#         variance.append(np.var(sorted[:val]))
#         std.append(np.std(sorted[:val]))

# plt.plot(variance, color='blue')
# plt.show()


### PLOT ###
def plot():
    sns.displot(df[args.metric], kde=True, color='blue')
    if args.overlap:
        sns.displot(df_all[args.metric], kde=True, color='red')
    # plt.xlim(0, args.xlim)
    plt.title(f'{args.lang} - {args.metric}_{args.quality}')
    plt.tight_layout()
    # plt.savefig(f'plots/{args.lang}_{args.partition}.png')
    # if args.metric != args.partition:
    #     if args.overlap and args.sample:
    #         plt.savefig(f'plots/{args.lang}_{args.metric}_calc_on_{args.partition}_{args.quality}_overlap_sampled.png')
    #     elif args.overlap:
    #         plt.savefig(f'plots/{args.lang}_{args.metric}_calc_on_{args.partition}_{args.quality}_overlap.png')
    #     else:
    #         plt.savefig(f'plots/{args.lang}_{args.metric}_calc_on_{args.partition}_{args.quality}.png')
    # else:
    #     if args.overlap and args.sample:
    #         plt.savefig(f'plots/{args.lang}_{args.partition}_{args.quality}_overlap_sampled.png')
    #     elif args.overlap:
    #         plt.savefig(f'plots/{args.lang}_{args.partition}_{args.quality}_overlap.png')
    #     else:
    #         plt.savefig(f'plots/{args.lang}_{args.partition}_{args.quality}.png')
    plt.show()

plot()