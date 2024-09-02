# Wikipedia Quality Estimation

wqe (Wikipedia Quality Estimation) is a simple toolkit for working with 
Wikipedia data in Python. It allows for loading and processing of Wikipedia
articles across all supported languages, as well as selecting high quality
data for training and evaluation of machine learning models.

## Installation

To install the package, simply run:

```
git clone https://github.com/akulmizev/Wikipedia_Quality_Estimation
cd Wikipedia_Quality_Estimation
pip install . 
```

If you would like to install the package in development mode, run:

```
pip install -e .
```

## Example Usage in Python

### Load and process the Hausa Wikipedia

```python
from wqe import WikiLoader, GOPHER_THRESHOLDS

# Load the Hausa wiki
wiki = WikiLoader('ha')
wiki.load_dataset()

# Filter out non-official scripts
wiki.pre_filter(script_regex=True)

# Deduplicate via exact match and min-hash (3-grams)
wiki.deduplicate(
    exact_match=True, 
    minhash=True, 
    jaccard_threshold=0.8,
    n_shingles=3
)

# Apply Gopher quality signal thresholds
wiki.apply_threshold(GOPHER_THRESHOLDS)

# Partition documents according to the longest articles 
wiki.apply_partition(
    split_method="balanced_chars", 
    metric="length_chars",
    quality=True
)
```
This will load the Hausa wiki via `wikimedia/wikipedia`
on the huggingface `datasets` hub, filter out not-official scripts,
deduplicate the articles, apply GOPHER thresholds, and partition the
articles according to the longest articles. 

The output should look like this (note that partitioning by `balanced_chars` will split the data in half):
```
INFO:wqe.data.loader:Loaded 36492 articles with 74084110 characters (train). Wiki: ha
INFO:wqe.data.processing:Filtering documents for accepted scripts: Latn
INFO:wqe.data.utils:Deleted: 25 docs (0.07%), 1785514 chars (2.41%), 1.77 MB (2.49%)
INFO:wqe.data.processing:Deduplicating dataset.
WARNING:wqe.data.processing:No tokenizer specified. Splitting on whitespace for minhash.
INFO:wqe.data.utils:Deleted: 1831 docs (5.02%), 745601 chars (1.03%), 0.72 MB (1.03%)
INFO:wqe.data.processing:Thresholding dataset by length_words, doc_mean_word_length, frac_lines_end_ellipsis, frac_symbol_to_words, frac_no_script_words...
INFO:wqe.data.utils:Deleted: 12315 docs (35.56%), 12839153 chars (17.94%), 12.35 MB (17.95%)
INFO:wqe.data.processing:Partition splitting method set to 'balanced_chars'.
INFO:wqe.data.processing:Partitioning dataset by length_chars...
INFO:wqe.data.utils:Deleted: 19420 docs (87.00%), 29355328 chars (50.00%), 28.22 MB (49.98%)
```

Currently supported partition functions are:

- `length`: article length in characters
- `unique_trigrams`: number of unique trigrams in the article
- `unique_words`: number of unique words
- `unique_chars`: number of unique characters
- `unique_character_trigrams`: number of unique character trigrams
- `unique_subwords`: number of unique subwords (tokenizer required)
- `unique_subword_trigrams`: number of unique subword trigrams (tokenizer required)
- `alpha_chars`: number of alphabetic characters in the article (`[a-zA-Z]`)

To see which languages are supported, call:

```python
from wqe import WikiLoader

WikiLoader.show_available_wikis()
```

... which outputs:

```commandline
Wiki ID        Language                                639-3     Scripts                       
----------------------------------------------------------------------------------------
ab             Abkhazian                               abk       Cyrl                          
ace            Achinese                                ace       Latn, Arab                    
ady            Adyghe                                  ady       Cyrl                          
af             Afrikaans                               afr       Latn                          
...
diq            Zazaki                                  diq       Latn                          
zea            Zeelandic                               zea       Latn                          
za             Zhuang                                  zha       Hani, Latn                    
zu             Zulu                                    zul       Latn                    
```

For details about the Python usage, see `./docs`. 

## Command Line Interface

wqe also provides a `hydra`-powered command line interface (CLI) for working with Wikipedia data. 
To load, process, and partition the Hausa Wikipedia, run:

```commandline
wqe --config-dir ./config/wqe +experiment=basic +dataset=basic
```

This assumes a directory structure like the following:

```
config/wqe
├── dataset
│   └── basic.yaml
├── experiment
│   └── basic.yaml
├── finetune
│   └── basic.yaml
├── pretrain
│   └── basic.yaml
└── tokenizer
    └── basic.yaml
```

... where `basic.yaml` is a configuration file for the respective task to be run, e.g.:

```yaml
# config/wqe/dataset/basic.yaml

pre_filter:
  script_regex: false
  lang_id: false
  char_cutoff: 100

partition:
  method: "balanced_chars"
  metric: "length"
  join_method: "intersection"

split:
  test_size: 0.1
  seed: 42
  shuffle: true

export: true
push_to_hub: true
```

The CLI assumes by default that the `experiment` config is provided, which contains
high-level settings for the experiment, e.g.:

```yaml
# config/wqe/experiment/basic.yaml

wiki_id: "ha"
experiment_id: "my_experiment"
wandb_entity: "wikiquality" #optional
local_path: "./experiments" #optional
hub_path: "WikiQuality" #optional
```

Given the provided config files, it is possible to load, process, and partition
a wiki dataset, train a tokenizer on it, and pass the components to `transformers`
for further language model pre-training and/or fine-tuning:

```commandline
wqe --config-dir ./config/wqe +experiment=basic +dataset=basic +tokenizer=basic +pretrain=basic +finetune=basic
```

To avoid generating separate config files for slight variations of an experiment,
it is likewise possible to pass overriding arguments directly to the CLI, e.g.:

```commandline
wqe --config-dir ./config/wqe \
+experiment=basic experiment.wiki_id="zu" \
+dataset=basic dataset.partition.metric="unique_trigrams" \
+tokenizer=basic tokenizer.vocab_size=10000 \
+pretrain=basic pretrain.checkpoint=True
+finetune=basic
```

All arguments are type-checked in `wqe.utils.config` and processed via 
`wqe.experiment.experiment.ExperimentRunner`. For more details, see `./docs`.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
