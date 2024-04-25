import numpy as np
from tokenizers.pre_tokenizers import Whitespace
# from tokenizers.pre_tokenizers import UnicodeScripts, Sequence, ByteLevel, WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from nltk.util import ngrams


class Partition:
    def __init__(self, config):
        # config is now passed from within the WikiDatasetFromConfig class
        self.__dict__.update(config.__dict__)
        self.higher_is_better = False
        if config.tokenizer:
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.tokenizer)

    def __call__(self, dataset):

        """
        This method partitions the dataset into two partitions based on the metric.
        :param dataset: A dataset object

        :return: A dataset object

        """

        metric_per_doc = [self.metric(item) for item in dataset["text"]]

        if self.method == "mean_cutoff":
            mean_cutoff = np.mean(metric_per_doc)  # + np.std(metric_per_doc)
            partition_1 = np.where(metric_per_doc < mean_cutoff)[0]
            partition_2 = np.where(metric_per_doc >= mean_cutoff)[0]

        elif self.method == "balanced_docs":
            half_point = len(dataset) // 2
            partition_1 = np.argsort(metric_per_doc)[:half_point]
            partition_2 = np.argsort(metric_per_doc)[half_point:]

        elif self.method == "balanced_chars":
            sorted_indices = np.argsort(metric_per_doc).tolist()
            char_budget = len("".join(dataset["text"])) // 2
            char_counter = 0
            partition_1 = []
            while char_counter < char_budget:
                char_counter += len(dataset[sorted_indices[0]]["text"])
                partition_1.append(sorted_indices.pop(0))
            partition_2 = sorted_indices

        else:
            raise ValueError("Partition method not recognized.")

        if (self.higher_is_better and self.quality) or (not self.higher_is_better and not self.quality):
            return partition_2
        else:
            return partition_1

    def metric(self, example):
        raise NotImplementedError("Metric not implemented. Please use a subclass.")

    def output_stats(self, **kwargs):
        pass


class Length(Partition):
    def __init__(self, config):
        super().__init__(config)
        self.higher_is_better = True

    def metric(self, example):
        return len(example)


class UniqueSubwords(Partition):
    def __init__(self, config):
        if not config.tokenizer:
            raise ValueError("Pass a tokenizer for this metric.")
        super().__init__(config)
        self.higher_is_better = True

    def metric(self, example):
        tokens = self.tokenizer.tokenize(example)
        return len(set(tokens))


class UniqueSubwordTrigrams(Partition):
    def __init__(self, config):
        if not config.tokenizer:
            raise ValueError("Pass a tokenizer for this metric.")
        super().__init__(config)
        self.higher_is_better = True

    def metric(self, example):
        tokens = self.tokenizer.tokenize(example)
        trigrams = list(ngrams(tokens, 3))
        return len(set(trigrams))


class UniqueTrigrams(Partition):
    def __init__(self, config):
        super().__init__(config)
        self.higher_is_better = True

    def metric(self, example):
        words = [word[0] for word in Whitespace().pre_tokenize_str(example)]
        trigrams = list(ngrams(words, 3))
        return len(set(trigrams))


class UniqueWords(Partition):
    def __init__(self, config):
        super().__init__(config)
        self.higher_is_better = True

    def metric(self, example):
        words = [word[0] for word in Whitespace().pre_tokenize_str(example)]
        return len(set(words))


class UniqueCharacters(Partition):
    def __init__(self, config):
        super().__init__(config)
        self.higher_is_better = True

    def metric(self, example):
        return len(set(example))


class UniqueCharacterTrigrams(Partition):
    def __init__(self, config):
        super().__init__(config)
        self.higher_is_better = True

    def metric(self, example):
        trigrams = list(ngrams(example, 3))
        return len(set(trigrams))


class AlphaChars(Partition):
    def __init__(self, config):
        super().__init__(config)
        self.higher_is_better = False

    def metric(self, example):
        return sum([1 for char in example if char.isalpha()])
