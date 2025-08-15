import dask.dataframe as dd
import pandas as pd # Keep pandas import for dtype validation and potential fallbacks

def group_by_aggregate(df, group_by_cols, agg_dict):
    """
    Performs a groupby operation with multiple aggregations on a Dask DataFrame.

    Technical: This function leverages Dask's groupby() method and the .agg() method,
    which can take a dictionary to perform different aggregations on different columns.
    It includes validation to ensure columns exist and that numeric aggregations
    are not attempted on non-numeric columns.

    Layman: This helps you create complex summaries of your data. For example, you can
    group by 'Department' and simultaneously calculate the total 'Salary' and the
    average 'Years of Service' for each department in one go.

    Args:
        df (dd.DataFrame): The input Dask DataFrame.
        group_by_cols (list): List of columns to group by.
        agg_dict (dict): A dictionary where keys are columns to aggregate and
                         values are the aggregation functions (e.g., {'col1': 'sum'}).

    Returns:
        dd.DataFrame: The aggregated Dask DataFrame.
        str: An error message if the operation fails, otherwise None.
    """
    if not isinstance(df, dd.DataFrame):
        return None, "Error: Invalid input Dask DataFrame."
    if not all(col in df.columns for col in group_by_cols):
        return None, "Error: One or more group by columns not found."
    if not isinstance(agg_dict, dict) or not agg_dict:
        return None, "Error: Aggregation dictionary must be a non-empty dictionary."

    # Validate aggregation columns and functions using Dask's meta information
    for col, func in agg_dict.items():
        if col not in df.columns:
            return None, f"Error: Aggregation column '{col}' not found."
        # Dask doesn't have a direct `is_numeric_dtype` method, so we check the metadata
        if func in ['mean', 'median', 'sum', 'std', 'var'] and not pd.api.types.is_numeric_dtype(df[col].dtype):
            return None, f"Error: Cannot apply numeric function '{func}' on non-numeric column '{col}'."

    try:
        aggregated_df = df.groupby(group_by_cols).agg(agg_dict).reset_index()
        # Dask's groupby.agg can create a single-level index, so the pandas flattening is not always needed, but it's good to be aware of.
        # For now, we rely on Dask's default behavior.
        return aggregated_df, None
    except Exception as e:
        return None, f"An unexpected error occurred during aggregation: {e}"

def create_pivot_table(df, index, columns, values, aggfunc):
    """
    Creates a pivot table from a Dask DataFrame.

    Technical: This is a wrapper around the Dask `pivot_table` function. It allows
    for reshaping the data by specifying which columns should be the new index,
    which should be the new columns, and which should be the values to aggregate.

    Layman: This function lets you restructure your data to see it from a different
    angle. For example, you could take a sales list and turn it into a table showing
    'Products' down the side, 'Regions' across the top, and total 'Sales' in the cells.

    Args:
        df (dd.DataFrame): The input Dask DataFrame.
        index (list): List of column names to use as the pivot table index.
        columns (list): List of column names to use as the pivot table columns.
        values (list): List of column names to aggregate.
        aggfunc (str or dict): The aggregation function(s) to use.

    Returns:
        dd.DataFrame: The resulting pivot table.
        str: An error message if the operation fails, otherwise None.
    """
    if not all(col in df.columns for col in index + columns + values):
        return None, "Error: One or more specified columns for the pivot table do not exist."

    try:
        pivot_df = dd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)
        return pivot_df.reset_index(), None
    except Exception as e:
        return None, f"An unexpected error occurred during pivot table creation: {e}"