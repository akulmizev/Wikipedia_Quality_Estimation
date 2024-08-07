### Train a tokenizer on wiki data:

```python
from wqe import WikiLoader, TokenizerConfig, HfTokenizerFromConfig

# Load Wikipedia data
wiki = WikiLoader("ha").load_dataset()

# Create a tokenization configuration
tokenizer_config = TokenizerConfig(
    model={"type": "unigram"},
    trainer={"type": "unigram"},
    normalizer={"type": "nkfc"},
    pre_tokenizer=[
        {"type": "whitespace"},
        {"type": "digits"},
        {"type": "metaspace", "prepend_scheme": "always", "replacement": "▁"}
    ],
    decoder={"type": "metaspace"},
    vocab_size=10000
)

# Train tokenization
tokenizer = HfTokenizerFromConfig.train_from_config(wiki, tokenizer_config)

# Save tokenization
tokenizer.save_pretrained("./models/unigram_tokenizer")
```
This will train a (fast) UnigramLM tokenizer with full compatibility with
the popular `PreTrainedTokenizerFast` class from `transformers`. Note: it is
possible to set the `vocab_size` to `auto` to automatically determine the
optimal vocabulary size based on the data (via Heap's Law and additional heuristics).
