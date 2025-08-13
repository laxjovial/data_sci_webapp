import pandas as pd

def group_by_aggregate(df, group_by_cols, agg_col, agg_func):
    """
    Performs a groupby and aggregation.

    Technical: This function uses pandas' groupby() method, selects a specific
    column for aggregation, and applies a specified aggregation function
    (like 'mean', 'sum', 'count'). It includes error handling for cases where
    a numeric aggregation is attempted on a non-numeric column.

    Layman: This helps you summarize your data. For example, you can group all
    your data by 'Country' and then find the average 'Salary' for each country.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_by_cols (list): List of columns to group by.
        agg_col (str): The column to aggregate.
        agg_func (str): The aggregation function ('mean', 'sum', 'count', etc.).

    Returns:
        pd.DataFrame: The aggregated DataFrame.
        str: An error message if the operation fails, otherwise None.
    """
    if not isinstance(df, pd.DataFrame):
        return None, "Error: Invalid input DataFrame."
    if not all(col in df.columns for col in group_by_cols):
        return None, "Error: One or more group by columns not found."
    if agg_col not in df.columns:
        return None, f"Error: Aggregation column '{agg_col}' not found."

    try:
        if agg_func in ['mean', 'median', 'sum', 'std']:
            # Check if the column is numeric before performing numeric aggregation
            if not pd.api.types.is_numeric_dtype(df[agg_col]):
                return None, f"Error: Cannot apply '{agg_func}' on non-numeric column '{agg_col}'."
            aggregated_df = df.groupby(group_by_cols)[agg_col].agg(agg_func).reset_index()
        elif agg_func in ['count', 'size']:
            aggregated_df = df.groupby(group_by_cols).size().reset_index(name='count')
        else:
            return None, f"Error: Invalid aggregation function '{agg_func}'."

        return aggregated_df, None
    except Exception as e:
        return None, f"An unexpected error occurred during aggregation: {e}"
