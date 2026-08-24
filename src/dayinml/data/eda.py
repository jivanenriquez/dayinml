"""
Functions for a first-look exploration of a dataset.

Intended for use in a Jupyter notebook
"""

import pandas as pd
from IPython.display import display


def explore(df):
    """Display a standard first-look summary of a dataframe

    For use in a Jupyter notebook. Shows head, shape, info, missing value
    counts and percentages, duplicate row count, categorical cardinality,
    and describe (split into numerical and categorical columns).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    """

    with pd.option_context("display.max_columns", None, "display.max_rows", None):
        print("FIRST 5 ROWS")
        display(df.head(5))

        print("\nSHAPE")
        print(df.shape)

        print("\nINFO")
        df.info()

        print("\nMISSING VALUES")
        missing_count = df.isna().sum()
        missing_count = missing_count[missing_count > 0].sort_values(ascending=False)
        if missing_count.empty:
            print("none")
        else:
            missing_pct = (missing_count / len(df) * 100).round(2)
            display(pd.DataFrame({"count": missing_count, "pct": missing_pct}))

        print("\nDUPLICATE ROWS")
        print(df.duplicated().sum())

        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(exclude="number").columns

        print("\nCARDINALITY (categorical)")
        if len(cat_cols):
            display(df[cat_cols].nunique().sort_values(ascending=False))
        else:
            print("none")

        print("\nDESCRIBE (numerical)")
        display(df[num_cols].describe() if len(num_cols) else "none")

        print("\nDESCRIBE (categorical)")
        display(df[cat_cols].describe() if len(cat_cols) else "none")

def varying_columns(df, keys):
    """Columns that differ within duplicate `keys` groups.

    Use when a key you believe identifies a row does not. Tells you whether the
    key is wrong (many columns vary) or the data is dirty (one or two vary).

    Parameters
    ----------
    df : DataFrame
    keys : str or list of str
        The candidate key.

    Returns
    -------
    Series
        Column name -> number of duplicate groups in which it varies,
        descending. Empty if `keys` is already unique.
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    dupes = df[df.duplicated(keys, keep=False)]
    if dupes.empty:
        return pd.Series(dtype='int64')

    n_unique = dupes.groupby(keys).nunique(dropna=False)
    varying = (n_unique > 1).sum()
    return varying[varying > 0].sort_values(ascending=False)
