import numpy as np

from sklearn.metrics import classification_report, precision_recall_fscore_support

class LossLogger:
    def __init__(self, n_samples, increment_by=1):
        self.n_samples = n_samples
        self.increment_by = increment_by
        self.counter = increment_by
        self.metrics_by_batch = []
        self.metrics_by_step = []

    def __call__(self, predictions):
        self.metrics_by_batch.extend([predictions])

    def _get_metric(self):
        return sum(self.metrics_by_batch) / self.n_samples

    def get_metric(self):

        metric_by_step = self._get_metric()
        self.metrics_by_step.append([self.counter] + [metric_by_step])
        self.counter += self.increment_by

        self.metrics_by_batch = []

        return metric_by_step

    def output_stats(self, path):
        with open(path, "w") as f:
            f.write("Step\tLoss\n")
            for stats in self.metrics_by_step:
                output = "\t".join(map(str, stats)) + "\n"
                f.write(output)

class PrecisionRecallF1Logger(LossLogger):
    def __init__(self, n_samples, increment_by=1):
        super().__init__(n_samples, increment_by)

    def __call__(self, predictions, labels):
        self.metrics_by_batch.extend(
            [precision_recall_fscore_support(labels, predictions, average="macro")]
        )

    def _get_metric(self):
        return precision_recall_fscore_support(np.array(self.metrics_by_batch))

    def output_stats(self, path):
        with open(path, "w") as f:
            f.write("Step\tPrecision\tRecall\tF1\n")
            for stats in self.metrics_by_step:
                output = "\t".join(map(str, stats)) + "\n"
                f.write(output)