import logging

import numpy as np

from datasets import load_dataset, DatasetDict
from datasets.exceptions import DatasetNotFoundError

from ..data.loader import WikiID

logger = logging.getLogger(__name__)


def validate_and_format_dataset(
        load_path: str,
        wiki: str,
        task: str
) -> DatasetDict:

    """
    Loads and validates a dataset for a specific task and language.
    First tries to load dataset by wiki id, then by alpha3 code, raising a ValueError otherwise.
    Then, checks if the dataset has the required columns for the task:
    - If a tagging dataset has a `ner_tags` or `upos` column, it is renamed to `tags`.
    - if a classification dataset has a `label` column, it is renamed to `labels`.

    Parameters
    ----------
    load_path : str
        The path to the dataset to load locally or from the huggingface hub.
    wiki : str
        The language identifier for the dataset being loaded (e.g. 'en').
    task : str
        The task for the dataset being loaded (e.g. 'ner').

    Returns
    -------
    DatasetDict
        The loaded and formatted dataset.
    """

    wiki = WikiID(wiki)

    try:
        logger.info(f"Attempting to load tagging dataset at {load_path} by wiki id: {wiki.id}.")
        dataset = load_dataset(load_path, wiki.id, trust_remote_code=True)
    except ValueError:
        try:
            logger.warning(f"Dataset not found. Attempting to load dataset by alpha3 code: {wiki.alpha3}.")
            dataset = load_dataset(load_path, wiki.alpha3, trust_remote_code=True)
        except (DatasetNotFoundError, FileNotFoundError, ValueError):
            raise ValueError(f"Could not find language-specific partition.\n"
                             f"Tried `load_dataset({load_path}, {wiki.id})`\n"
                             f"and `load_dataset({load_path}, {wiki.alpha3}).`\n")

    logger.info(f"Dataset loaded successfully.")

    assert "train" in dataset.keys(), "Train split must be present in the dataset."

    if task in ["pos", "ner"]:
        assert any(feature in dataset["train"].features for feature in ["tags", "ner_tags", "upos"]), \
            "Dataset must have a `tags`, `ner_tags`, or `upos` column formatted as features."
        for column in ["ner_tags", "upos"]:
            if column in dataset["train"].features:
                dataset = dataset.rename_column(column, "tags")
    else:
        assert ("label" in dataset["train"].features) or ("labels" in dataset["train"].features), \
            "Dataset must have a `label` column formatted as features."
        if "label" in dataset["train"].features:
            dataset = dataset.rename_column("label", "labels")
        if "tweet" in dataset["train"].features:
            dataset = dataset.rename_column("tweet", "text")

    return dataset


def np_encoder(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):

        return int(obj)

    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)

    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}

    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    elif isinstance(obj, (np.bool_)):
        return bool(obj)

    elif isinstance(obj, (np.void)): 
        return None

    print(f'Cannot parse {obj} into json, using None instead!')
    return None
