import logging

import numpy as np

from datasets import load_dataset, DatasetDict, ClassLabel, Sequence
from datasets.exceptions import DatasetNotFoundError

from ..data.loader import WikiID

logger = logging.getLogger(__name__)


def validate_and_format_dataset(
        load_path: str,
        wiki: str,
        task: str,
        columns: list,
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
    columns: list
        The column names for the text and label.

    Returns
    -------
    DatasetDict
        The loaded and formatted dataset.
    """

    wiki = WikiID(wiki)

    try:
        logger.info(f"Attempting to load dataset at {load_path} by wiki id: {wiki.id}.")
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
        if columns:
            if columns[0] != 'text':
                dataset = dataset.rename_column(columns[0], "text")
            if columns[1] != 'tags':
                dataset = dataset.rename_column(columns[1], "tags")
            logger.info('Renaming columns to "text" and "tags"')
        else:
            assert any(feature in dataset["train"].features for feature in ["tags", "ner_tags", "upos"]), \
                "Dataset must have a `tags`, `ner_tags`, or `upos` column formatted as features."
            for column in ["ner_tags", "upos"]:
                if column in dataset["train"].features:
                    dataset = dataset.rename_column(column, "tags")
    elif task == "nli":
        if columns:
            if columns[0] != "premise":
                dataset = dataset.rename_column(columns[0], "premise")
            if columns[1] != "hypothesis":
                dataset = dataset.rename_column(columns[0], "hypothesis")
            if columns[2] != "labels":
                dataset = dataset.rename_column(columns[0], "labels")
        else:
            assert all(feature in dataset["train"].features for feature in ["premise", "hypothesis"]), \
                "Dataset must have `premise` and `hypothesis` columns formatted as features."
            if "label" in dataset["train"].features:
                dataset = dataset.rename_column("label", "labels")
    else:
        if columns:
            if columns[0] != 'text':
                dataset = dataset.rename_column(columns[0], "text")
            if columns[1] != 'labels':
                dataset = dataset.rename_column(columns[1], "labels")
            logger.info('Renaming columns to "text" and "labels"')
        else:
            assert ("label" in dataset["train"].features) or ("labels" in dataset["train"].features), \
                "Dataset must have a `label` column formatted as features."
            if "label" in dataset["train"].features:
                dataset = dataset.rename_column("label", "labels")
            if "tweet" in dataset["train"].features:
                dataset = dataset.rename_column("tweet", "text")

    return dataset


def np_encoder(obj):
    """
    Make sure numpy objects are json serializable.
    Taken from: https://stackoverflow.com/a/61903895
    """
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

def validate_and_format_splits(
        train_path: str,
        valid_path: str,
        test_path: str,
        wiki: str,
        task: str,
        columns: list,
) -> DatasetDict:

    """
    Loads and validates a dataset split for a specific task and language.
    First tries to load dataset as is, then by wiki id, and then by alpha3 code, raising a ValueError otherwise.
    

    Parameters
    ----------
    train_path : str
        The path to the train split to load locally or from the huggingface hub.
    valid_path: str
        The path to the validation split to load locally or from the huggingface hub.
    test_path: str
        The path to the test split to load locally or from the huggingface hub.
    wiki : str
        The language identifier for the dataset being loaded (e.g. 'en').
    task : str
        The task for the dataset being loaded (e.g. 'ner').
    columns: list
        The column names for the text and label.


    Returns
    -------
    DatasetDict
        The loaded and formatted dataset.
    """
 
    wiki = WikiID(wiki)
    
    assert train_path, "A path to the train split must be provided."
    assert valid_path, "A path to the validation split must be provided."
    assert test_path, "A path to the test split must be provided."
    # assert columns, "Please provide column names for the text and labels."

    try:
        logger.info(f"Attempting to load train split at {train_path}")
        train = load_dataset(train_path, split="train")
    except:
        try:
            logger.info(f"Dataset not found. Attempting to load dataset at {train_path} by wiki id: {wiki.id}.")
            train = load_dataset(train_path, wiki.id, split="train", trust_remote_code=True)
        except ValueError:
            try:
                logger.warning(f"Dataset not found. Attempting to load dataset by alpha3 code: {wiki.alpha3}.")
                train = load_dataset(train_path, wiki.alpha3, split="train", trust_remote_code=True)
            except (DatasetNotFoundError, FileNotFoundError, ValueError):
                raise ValueError(f"Could not find dataset.")
    
    try:
        logger.info(f"Attempting to load validation split at {valid_path}")
        valid = load_dataset(valid_path, split="validation")
    except:
        try:
            logger.info(f"Dataset not found. Attempting to load dataset at {valid_path} by wiki id: {wiki.id}.")
            valid = load_dataset(valid_path, wiki.id, split="validation", trust_remote_code=True)
        except ValueError:
            try:
                logger.warning(f"Dataset not found. Attempting to load dataset by alpha3 code: {wiki.alpha3}.")
                valid = load_dataset(valid_path, wiki.alpha3, split="validation", trust_remote_code=True)
            except (DatasetNotFoundError, FileNotFoundError, ValueError):
                raise ValueError(f"Could not find dataset.")
            
    try:
        logger.info(f"Attempting to load test split at {test_path}")
        test = load_dataset(test_path, split="test")
    except:
        try:
            logger.info(f"Dataset not found. Attempting to load dataset at {test_path} by wiki id: {wiki.id}.")
            test = load_dataset(test_path, wiki.id, split="test", trust_remote_code=True)
        except ValueError:
            try:
                logger.warning(f"Dataset not found. Attempting to load dataset by alpha3 code: {wiki.alpha3}.")
                test = load_dataset(test_path, wiki.alpha3, split="test", trust_remote_code=True)
            except (DatasetNotFoundError, FileNotFoundError, ValueError):
                raise ValueError(f"Could not find dataset.")
    
    logger.info(f"Splits loaded successfully.")

    if task in ["pos", "ner"]:
        if columns:
            if len(columns) == 2:
                if columns[0] != 'tokens':
                    train = train.rename_column(columns[0], "tokens")
                    valid = valid.rename_column(columns[0], "tokens")
                    test = test.rename_column(columns[0], "tokens")
                if columns[1] != 'tags':
                    train = train.rename_column(columns[0], "tags")
                    valid = valid.rename_column(columns[0], "tags")
                    test = test.rename_column(columns[0], "tags")
                logger.info('Renaming columns to "text" and "tags"')
            else:
                assert len(columns) == 3, """If text and tag columns have the same name across train, valid and test datasets, \n
                                            specify column names as [<text_column_name>, (label_column_name>)]. \n
                                            If they are dfferent then specify column names as: \n
                                            [[train column name list], [valid column name list], [test column name list]]."""
                train_columns = columns[0]
                if train_columns[0] != 'tokens':
                    train = train.rename_column(train_columns[0], "tokens")
                if train_columns[1] != 'tags':
                    train = train.rename_column(train_columns[1], "tags")

                valid_columns = columns[1]
                if valid_columns[0] != 'tokens':
                    valid = valid.rename_column(valid_columns[0], "tokens")
                if valid_columns[1] != 'tags':
                    valid = valid.rename_column(valid_columns[1], "tags")

                test_columns = columns[2]
                if test_columns[0] != 'tokens':
                    test = train.rename_column(test_columns[0], "tokens")
                if test_columns[1] != 'tags':
                    test = train.rename_column(test_columns[1], "tags")
                
                
        else:
            assert 'tokens' in train.features, "Train split must have a `text` column formatted as features."
            assert 'tokens' in valid.features, "Validation split must have a `text` column formatted as features."
            assert 'tokens' in test.features, "Test split must have a `text` column formatted as features."
            assert any(feature in train.features for feature in ["tags", "ner_tags", "upos"]), \
                "Train split must have a `tags`, `ner_tags`, or `upos` column formatted as features."
            assert any(feature in valid.features for feature in ["tags", "ner_tags", "upos"]), \
                "Validation split must have a `tags`, `ner_tags`, or `upos` column formatted as features."
            assert any(feature in test.features for feature in ["tags", "ner_tags", "upos"]), \
                "Test split must have a `tags`, `ner_tags`, or `upos` column formatted as features."
            
            for column in ["ner_tags", "upos"]:
                if column in train.features:
                    train = train.rename_column(column, "tags")
                if column in valid.features:
                    valid = valid.rename_column(column, "tags")
                if column in test.features:
                    test = test.rename_column(column, "tags")
    

    elif task == "nli":
        if columns:
            if len(columns) == 3:
                if columns[0] != "premise":
                    train = train.rename_column(columns[0], "premise")
                    valid = valid.rename_column(columns[0], "premise")
                    test = test.rename_column(columns[0], "premise")
                if columns[1] != "hypothesis":
                    train = train.rename_column(columns[0], "hypothesis")
                    valid = valid.rename_column(columns[0], "hypothesis")
                    test = test.rename_column(columns[0], "hypothesis")
                if columns[2] != "labels":
                    train = train.rename_column(columns[0], "labels")
                    valid = valid.rename_column(columns[0], "labels")
                    test = test.rename_column(columns[0], "labels")
            else:
                assert len(columns) == 3, """If premise, hypothesis and labels columns have the same name across train, valid and test datasets, \n
                                            specify column names as [<premise_column_name>, <hypothesis_column_nmae>, <label_column_name>]. \n
                                            If they are dfferent then specify column names as: \n
                                            [[train column name list], [valid column name list], [test column name list]]."""
                
                train_columns = columns[0]
                if train_columns[0] != 'premise':
                    train = train.rename_column(train_columns[0], "premise")
                if train_columns[1] != 'hypothesis':
                    train = train.rename_column(train_columns[1], "hypothesis")
                if train_columns[2] != 'labels':
                    train = train.rename_column(train_columns[3], "labels")

                valid_columns = columns[1]
                if valid_columns[0] != 'premise':
                    valid = valid.rename_column(valid_columns[0], "premise")
                if valid_columns[1] != 'hypothesis':
                    valid = valid.rename_column(valid_columns[1], "hypothesis")
                if valid_columns[2] != 'labels':
                    valid = valid.rename_column(valid_columns[3], "labels")

                test_columns = columns[2]
                if test_columns[0] != 'premise':
                    test = test.rename_column(test_columns[0], "premise")
                if test_columns[1] != 'hypothesis':
                    test = test.rename_column(test_columns[1], "hypothesis")
                if test_columns[2] != 'labels':
                    test = test.rename_column(test_columns[3], "labels")
        

        else:
            assert all(feature in train.features for feature in ["premise", "hypothesis"]), \
                "Train split must have `premise` and `hypothesis` columns formatted as features."
            assert all(feature in valid.features for feature in ["premise", "hypothesis"]), \
                "Validation split must have `premise` and `hypothesis` columns formatted as features."
            assert all(feature in test.features for feature in ["premise", "hypothesis"]), \
                "Test split must have `premise` and `hypothesis` columns formatted as features."
            
            assert ("label" in train.features) or ("labels" in train.features), \
                "Dataset must have a `label`, or `labels` column formatted as features."
            assert ("label" in valid.features) or ("labels" in train.features), \
                "Dataset must have a `label`, or `labels` column formatted as features."
            assert ("label" in test.features) or ("labels" in train.features), \
                "Dataset must have a `label`, or `labels` column formatted as features."
        
            if "label" in train.features:
                train = train.rename_column("label", "labels")
            if "label" in valid.features:
                valid = valid.rename_column("label", "labels")
            if "label" in test.features:
                test = test.rename_column("label", "labels")
    else:
        if columns:
            if len(columns) == 2:
                if columns[0] != 'text':
                    train = train.rename_column(columns[0], "text")
                    valid = valid.rename_column(columns[0], "text")
                    test = test.rename_column(columns[0], "text")
                if columns[1] != 'labels':
                    train = train.rename_column(columns[0], "labels")
                    valid = valid.rename_column(columns[0], "labels")
                    test = test.rename_column(columns[0], "labels")
                logger.info('Renaming columns to "text" and "labels"')
            else:
                assert len(columns) == 3, """If text and labels columns have the same name across train, valid and test datasets, \n
                                            specify column names as [<text_column_name>, <label_column_name>]. \n
                                            If they are dfferent then specify column names as: \n
                                            [[train column name list], [valid column name list], [test column name list]]."""
                train_columns = columns[0]
                if train_columns[0] != 'text':
                    train = train.rename_column(train_columns[0], "text")
                if train_columns[1] != 'labels':
                    train = train.rename_column(train_columns[1], "labels")

                valid_columns = columns[1]
                if valid_columns[0] != 'text':
                    valid = valid.rename_column(valid_columns[0], "text")
                if valid_columns[1] != 'labels':
                    valid = valid.rename_column(valid_columns[1], "labels")

                test_columns = columns[2]
                if test_columns[0] != 'text':
                    test = train.rename_column(test_columns[0], "text")
                if test_columns[1] != 'labels':
                    test = train.rename_column(test_columns[1], "labels")
                
                
        else:
            assert 'text' in train.features, "Train split must have a `text` column formatted as features."
            assert 'text' in valid.features, "Validation split must have a `text` column formatted as features."
            assert 'text' in test.features, "Test split must have a `text` column formatted as features."
            assert ("label" in train.features) or ("labels" in train.features), \
                "Dataset must have a `label`, or `labels` column formatted as features."
            assert ("label" in valid.features) or ("labels" in train.features), \
                "Dataset must have a `label`, or `labels` column formatted as features."
            assert ("label" in test.features) or ("labels" in train.features), \
                "Dataset must have a `label`, or `labels` column formatted as features."
            
            if "label" in train.features:
                train = train.rename_column("label", "labels")
            if "label" in valid.features:
                valid = valid.rename_column("label", "labels")
            if "label" in test.features:
                test = test.rename_column("label", "labels")

    dataset = DatasetDict({"train": train, "test": test, "valid": valid})
    if task in ["pos", "ner"]:    
        dataset = dataset.cast_column("tags", ClassLabel(num_classes=len(set(dataset['train']['tags']))))
    else:
        dataset = dataset.cast_column("labels", ClassLabel(num_classes=len(set(dataset['train']['labels']))))
    return dataset
