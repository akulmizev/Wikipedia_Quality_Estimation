### Train a masked language model on wiki data:

```python
from wqe import (
    WikiLoader,
    HfTokenizerFromConfig,
    TrainingParameters,
    MLM
)

# Load Wikipedia data
wiki = WikiLoader("ha").load_dataset()

# Load Tokenizer
tokenizer = HfTokenizerFromConfig.from_pretrained("./models/unigram_tokenizer")

# Specify training parameters
params = TrainingParameters(
    model_type="deberta",
    task="mlm",
    batch_size=8,
    max_length=512,
    num_train_epochs=3,
    lr=1e-3,
    num_eval_steps=100,
    padding_strategy="max_length"
)

# Initialize the model
deberta_model = MLM(
    load_path="./config/model/tiny_deberta/config.json",
    config=params
)

# Train the model
deberta_model.train(wiki, tokenizer)

# Save the model
deberta_model.save("./models/deberta_mlm")
```

This will train a masked language model (DeBERTa v1) on the selected Wikipedia data using
the `transformers` library and save it to the specified path.
If passing a `json` file to `load_path`, the model will be initialized from
the specified configuration file. Causal language models can also be trained by
setting `task="clm"` and passing a `TrainingParameters` object with the desired
configuration, e.g.:

```python
from wqe import CLM

# Initialize the model
params = params.model="gpt_neo"
params = params.task="clm"

tiny_gpt_neo = CLM(
    load_path="./config/model/tinystories/config.json",
    config=params
)
```