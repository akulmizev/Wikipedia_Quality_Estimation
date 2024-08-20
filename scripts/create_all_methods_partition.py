from datasets import load_dataset
from nltk.util import ngrams
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
from kneed import KneeLocator
import pandas as pd

def unique_trigrams(example: str) -> int:
    words = [word[0] for word in Whitespace().pre_tokenize_str(example)]
    trigrams = list(ngrams(words, 3))
    return len(set(trigrams))

def length(example: str) -> int:
    return len(example)

def unique_words(example: str) -> int:
    words = [word[0] for word in Whitespace().pre_tokenize_str(example)]
    return len(set(words))

def unique_characters(example: str) -> int:
    return len(set(example))

def unique_character_trigrams(example: str) -> int:
    trigrams = list(ngrams(example, 3))
    return len(set(trigrams))

function_map = {'unique_trigrams': unique_trigrams,
                'length': length,
                'unique_words': unique_words,
                'unique_characters': unique_characters,
                'unique_character_trigrams': unique_character_trigrams}

wiki = 'pcm'
data = load_dataset("WikiQuality/raw_wiki", wiki)

print(len(data['train']))
print(len(data['test']))
# for metric in function_map.keys():
#     data = data.map(lambda x: {metric: function_map[metric](x['text'])})
# df = pd.DataFrame(data['train'], columns=data['train'].features.keys())
# for metric in function_map.keys():
#     df[metric] = df[metric].astype(int)
#     arr = np.array(df[metric].tolist())
#     arr_norm = (arr-np.min(arr))/(np.max(arr)-np.min(arr)) * 100
#     df[metric] = list(arr_norm)

# df['All'] = df['unique_trigrams'] + df['length'] + df['unique_words'] + df['unique_characters'] + df['unique_character_trigrams']
# sorted = df['All'].sort_values(ascending=True).tolist()

# variances = [np.var(sorted[:i]) for i in range(1, len(sorted))]
# n_values = np.arange(1, len(sorted))
# knee = KneeLocator(n_values, variances, curve='convex', direction='increasing')

# df = df.sort_values(by='All', ascending=True)
# new_df = df[knee.knee:]
# print(knee.knee)
