"""
Functions for a first-look exploration of a dataset.

Intended for use in a Jupyter notebook
"""

import pandas as pd
from IPython.display import display


def explore(df):
    """EDA Starting Point always

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    """
    # Check dataset shape
    print(f"shape: {df.shape}")

    # Check for duplicates
    print('duplicate rows:', df.duplicated().sum())

    # Missing report
    print('\n--- missing values ---')
    missing = df.isnull().sum().loc[lambda x: x > 0].sort_values(ascending=False)
    missing_report = pd.DataFrame({
        'column': missing.index,
        'missing_count': missing.values,
        'missing_pct': (missing.values / len(df) * 100).round(2),
    }).reset_index(drop=True)
    display(missing_report)

    # Describe cat cols
    print('\n--- categorical columns ---')
    display(df.describe(include='object').T)

    # Describe num cols
    print('\n--- numerical columns ---')
    display(df.describe(include='number').T)

    # Columns describe doesn't cover
    uncovered = set(df.columns) - set(df.describe(include='number').columns) - set(df.describe(include='object').columns)
    # uncovered = set(df.columns[:10]) # TEST to see if IF condition works
    if uncovered:
        print('\n--- columns not covered by describe ---')
        display(df[list(uncovered)].describe(include='all').T)


    # Check sample rows (powered by the meaning of life)
    print('\n--- sample rows ---')
    with pd.option_context('display.max_columns', None):
        display(df.sample(5, random_state=42))
    

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


def positive_rate_by(df, by, target='label'):
    """Count, positive count and positive rate of a binary target within groups.

    Parameters
    ----------
    df : DataFrame
    by : str, list of str, or Series
        Grouping key. Pass a Series (for example the output of `pd.cut`) to
        group by buckets built in view of the reader.
    target : str, default 'label'
        Binary target column.

    Returns
    -------
    DataFrame
        Indexed by group, with columns `rows`, `positives` and `positive_rate`.
        `positive_rate` is a float, so the caller formats it at display time.
    """
    return df.groupby(by, observed=True)[target].agg(
        rows='size', positives='sum', positive_rate='mean')


def null_rate_by(df, by, min_rate=0.0):
    """Null share of every column within groups.

    Use to tell missingness that is a property of the rows from missingness
    that is a property of the group, such as a collection change in one year.

    Parameters
    ----------
    df : DataFrame
    by : str, list of str, or Series
        Grouping key.
    min_rate : float, default 0.0
        Keep only columns whose null share exceeds this in at least one group.
        Raise it to drop columns that are null in a handful of rows.

    Returns
    -------
    DataFrame
        Group by column, holding the null share as a float.
    """
    rates = df.groupby(by, observed=True).apply(
        lambda g: g.isna().mean(), include_groups=False)
    return rates.loc[:, rates.max() > min_rate]
