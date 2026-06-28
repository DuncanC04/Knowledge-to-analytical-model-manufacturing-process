"""Dataset loading and the extrapolative train / validation / test split.

The split reproduces the protocol from the paper: a low-value region of the
parameter space is used for training/validation, while the high-value region is
held out as an extrapolative test set.
"""

import pandas as pd

from .config_loader import resolve_path


def load_and_split(cfg: dict) -> dict:
    """Load the spreadsheet and build the train/validation/test split.

    Returns a dict:
        {
          "train":      (X_tuple, y_array),
          "validation": (X_tuple, y_array),
          "test":       (X_tuple, y_array),
          "train_df":   pandas.DataFrame,     # the raw training rows
        }
    where X_tuple is a tuple of NumPy arrays, one per input variable, in the
    order defined by `variables.inputs` in the config.
    """
    ds = cfg["dataset"]
    file_path = resolve_path(cfg, ds["file"])
    input_cols = cfg["input_columns"]
    output_col = cfg["output_column"]

    original_df = pd.read_excel(file_path)

    # Iteratively keep only the bottom `ratio` fraction by value for each
    # filter column -> this carves out the low-value train/validation pool.
    filter_cols = ds.get("filter_columns") or input_cols
    current_df = original_df.copy()
    for col in filter_cols:
        threshold = current_df[col].quantile(ds["ratio"])
        current_df = current_df[current_df[col] < threshold].copy()

    # Everything outside the low-value region is the extrapolative test set.
    test_data = original_df.drop(current_df.index)

    # Randomly sample the training rows; the remainder is validation.
    train_data = current_df.sample(n=ds["train_size"], random_state=ds["random_state"])
    validation_data = current_df.drop(train_data.index)

    def extract(df: pd.DataFrame):
        X = tuple(df[col].values for col in input_cols)
        y = df[output_col].values
        return X, y

    return {
        "train": extract(train_data),
        "validation": extract(validation_data),
        "test": extract(test_data),
        "train_df": train_data,
    }


def split_summary(data: dict) -> str:
    """A short, printable description of the split sizes."""
    return (
        f"Train: {len(data['train'][1])} rows | "
        f"Validation: {len(data['validation'][1])} rows | "
        f"Test: {len(data['test'][1])} rows"
    )
