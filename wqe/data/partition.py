import numpy as np
from datasets import Dataset
from tokenizers.pre_tokenizers import UnicodeScripts, Whitespace, Sequence, ByteLevel, WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from nltk.util import ngrams


class Partition:
    def __init__(self, config):
        # config is now passed from within the WikiDatasetFromConfig class
        self.config = config["partition"]
        self.type = self.config["partition_type"]

    def __call__(self, dataset):
        metric_per_doc = [self.metric(item) for item in dataset["text"]]
        if self.config["partition_type"] == "mean_cutoff":
            mean_cutoff = np.mean(metric_per_doc)  # + np.std(metric_per_doc)
            partition_1 = np.where(metric_per_doc < mean_cutoff)[0]
            partition_2 = np.where(metric_per_doc >= mean_cutoff)[0]
        elif self.config["partition_type"] == "balanced":
            half_point = len(dataset) // 2
            partition_1 = np.argsort(metric_per_doc)[:half_point]
            partition_2 = np.argsort(metric_per_doc)[half_point:]
        elif self.config["partition_type"] == "balanced_character_budget":
            text = dataset["text"]
            budget = sum(len(item) for item in text) / 2
            a_l = []
            total_num_chars = 0
            partition_1 = []
            partition_2 = []
            for i in range(len(text)):
                a_l.append((text[i], metric_per_doc[i]))
            a_lsorted = sorted(a_l, key=lambda x: x[1])
            for i, j in a_lsorted:
                if total_num_chars < budget:
                    partition_1.append(metric_per_doc.index(j))
                    total_num_chars += len(i)
                else:
                    partition_2.append(metric_per_doc.index(j))
        else:
            raise ValueError("Partition type not recognized.")

        # Not sure if this will work...
        if self.config["partition_metric"] != "all":
            if (self.config["higher_is_better"] and self.config["quality"]) or \
                    (not self.config["higher_is_better"] and not self.config["quality"]):
                dataset = dataset.select(partition_2)
            else:
                dataset = dataset.select(partition_1)
        else:
            dataset = dataset.select(partition_2)

        # elif (self.config["higher_is_better"] and not self.config["quality"]) \
        #         or (not self.config["higher_is_better"] and self.config["quality"]):
        #     dataset = dataset.select(partition_1)

        # if self.config["higher_is_better"]:
        #     if self.config["quality"]:
        #         dataset = dataset.select(partition_2)
        #     else:
        #         dataset = dataset.select(partition_1)
        # else:
        #     if self.config["quality"]:
        #         dataset = dataset.select(partition_1)
        #     else:
        #         dataset = dataset.select(partition_2)

        return dataset

    def metric(self, example):
        raise NotImplementedError("Metric not implemented. Please use a subclass.")

    def output_stats(self, **kwargs):
        pass


class Length(Partition):
    def __init__(self, config):
        super().__init__(config)

    def metric(self, example):
        return len(example)


class UniqueSubwords(Partition):
    def __init__(self, config):
        super().__init__(config)
        self.config = config["partition"]
        # self.tokenizer = tokenizer

    def metric(self, example):
        tokenizer_file = self.config["partition_tokenizer"]
        if tokenizer_file is False:
            raise ValueError("Pass a tokenizer for this metric.")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        tokens = tokenizer.tokenize(example)
        return len(set(tokens))


class UniqueSubwordTrigrams(Partition):
    def __init__(self, config):
        super().__init__(config)
        self.config = config["partition"]

    def metric(self, example):
        tokenizer_file = self.config["partition_tokenizer"]
        if tokenizer_file is False:
            raise ValueError("Pass a tokenizer for this metric.")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        tokens = tokenizer.tokenize(example)
        trigrams = list(ngrams(tokens, 3))
        return len(set(trigrams))


class UniqueTrigrams(Partition):
    def __init__(self, config):
        super().__init__(config)

    def metric(self, example):
        words = [word[0] for word in Whitespace().pre_tokenize_str(example)]
        trigrams = list(ngrams(words, 3))
        return len(set(trigrams))


class UniqueWords(Partition):
    def __init__(self, config):
        super().__init__(config)

    def metric(self, example):
        words = [word[0] for word in Whitespace().pre_tokenize_str(example)]
        return len(set(words))


class UniqueCharacters(Partition):
    def __init__(self, config):
        super().__init__(config)

    def metric(self, example):
        return len(set(example))


class UniqueCharacterTrigrams(Partition):
    def __init__(self, config):
        super().__init__(config)

    def metric(self, example):
        trigrams = list(ngrams(example, 3))
        return len(set(trigrams))


class AlphaChars(Partition):
    def __init__(self, config):
        super().__init__(config)

    def metric(self, example):
        return sum([1 for char in example if char.isalpha()])
