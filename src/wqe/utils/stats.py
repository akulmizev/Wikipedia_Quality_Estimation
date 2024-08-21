import logging

import numpy as np
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


def ks_statistic(sample, full_data):
    return stats.ks_2samp(sample, full_data).statistic


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))


def find_elbow(x, y):
    # Normalize data
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Find the point with maximum distance from the line
    distances = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0]) / np.sqrt(
        (y[-1] - y[0]) ** 2 + (x[-1] - x[0]) ** 2)
    elbow_index = np.argmax(distances)

    return elbow_index


def normalize(data):

    z_scores = stats.zscore(data)

    normalized = (z_scores - np.min(z_scores)) / (np.max(z_scores) - np.min(z_scores))

    return normalized


def get_representative_sample_size(distribution, metric="ks"):

    if metric not in ["ks", "js"]:
        logger.warning(f"Invalid metric '{metric}'. Using 'ks' instead.")
        metric = "ks"

    sample_sizes = np.logspace(1, np.log10(distribution.shape[0]), 25).astype(int)
    sample_sizes = np.unique(sample_sizes)  # Remove duplicates

    # Arrays to store results
    results = []

    # Calculate statistics for each sample size
    for n in tqdm(sample_sizes):
        metrics_n = []
        for _ in range(5):  # Repeat sampling 10 times for robustness
            sample = np.random.choice(distribution, size=n, replace=False)
            if metric == "ks":
                metrics_n.append(ks_statistic(sample, distribution))
            elif metric == "js":
                hist_sample, _ = np.histogram(sample, bins=100, density=True)
                hist_full, _ = np.histogram(distribution, bins=100, density=True)
                metrics_n.append(js_divergence(hist_sample, hist_full))

        results.append(np.mean(metrics_n))

    elbow_index = find_elbow(sample_sizes, results)

    return sample_sizes[elbow_index]

