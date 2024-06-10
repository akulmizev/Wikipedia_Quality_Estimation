### Fine-tune a trained model on a downstream task:

```python
from datasets import load_dataset
from wqe import FastTokenizerFromConfig, TrainingParameters, Classifier

# Load dataset
masakhanews = load_dataset("masakhanews", "hau")

# Load tokenizer
tokenizer = FastTokenizerFromConfig.from_pretrained("./models/unigram_tokenizer")

# Specify training parameters
params = TrainingParameters(
    model_type="deberta",
    task="classification",
    max_length=512,
    num_train_epochs=10,
    batch_size=32,
    lr=1e-3,
    padding_strategy="max_length"
)

# Initialize the model
deberta_classifier = Classifier(load_path="./models/deberta_mlm", config=params)

# Train the model
deberta_classifier.train(masakhanews, tokenizer, eval_split="validation")

# Test the model
deberta_classifier.test(masakhanews, split="test")
```

Currently supported tasks are:

- `classification`: text classification with `Classifier`
- `pos`: part-of-speech tagging with `Tagger`
- `ner`: named entity recognition with `Tagger`