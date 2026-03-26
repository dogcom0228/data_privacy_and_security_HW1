import csv
import numpy as np
import pandas as pd


def auto_cast(value):
    if value is None:
        return None
    v = value.strip()
    if v == "":
        return ""

    lower = v.lower()
    if lower in ("true", "false"):
        return lower == "true"

    try:
        return int(v)
    except ValueError:
        pass

    try:
        return float(v)
    except ValueError:
        return value


def read_csv(file_path, mode="r", encoding="utf-8"):
    """Reads a CSV file and returns a list of dictionaries with auto type casting."""
    with open(file_path, mode=mode, encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        return [{k: auto_cast(v) for k, v in row.items()} for row in reader]


def format_value(val):
    if val is None:
        return ""
    if isinstance(val, tuple):
        return "-".join(str(v) for v in val)
    return str(val)


def preprocess_adult(file_path: str, columns=None):
    """
    Read and preprocess adult dataset.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    columns : list[str] or None
        If provided, align one-hot columns to this list (for test-time consistency).

    Returns
    -------
    X : numpy.ndarray, shape (N, D), dtype float32
    y : numpy.ndarray, shape (N, 1), dtype float32
    columns : list[str]
        Column names after one-hot encoding.
    """
    records = read_csv(file_path)
    df = pd.DataFrame(records)
    df = df.dropna()

    df["income"] = (
        df["income"].astype(str).str.strip().str.rstrip(".") == ">50K"
    ).astype("float32")

    y = df["income"].values.reshape(-1, 1).astype(np.float32)
    X = df.drop(columns=["income"])

    X = pd.get_dummies(X, drop_first=True)

    # Align to training columns (add missing cols as 0, drop extra cols)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)

    return X.values.astype(np.float32), y, list(X.columns)
