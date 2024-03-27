import numpy as np


class Partition:
    def __init__(self, config):
        self.config = config["data"]["partition"]
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
        else:
            raise ValueError("Partition type not recognized.")
        if self.config["higher_is_better"]:
            return dataset[partition_2], dataset[partition_1]
        else:
            return dataset[partition_1], dataset[partition_2]

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

    def metric(self, example):
        raise NotImplementedError("Metric not implemented. Please use a subclass.")

class UniqueSubwordTrigrams(Partition):
    def __init__(self, config):
        super().__init__(config)

    def metric(self, example):
        raise NotImplementedError("Metric not implemented. Please use a subclass.")

class AlphaChars(Partition):
    def __init__(self, config):
        super().__init__(config)

    def metric(self, example):
        return sum([1 for char in example if char.isalpha()])