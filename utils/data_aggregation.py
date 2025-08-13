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
        str: An error message, if any.
    """
    error_message = None
    try:
        # Check if group by columns exist
        if not all(col in df.columns for col in group_by_cols):
            return None, "Error: One or more group by columns not found in DataFrame."

        # Check if aggregation column exists
        if agg_col not in df.columns:
            return None, f"Error: Aggregation column '{agg_col}' not found in DataFrame."
            
        # Check if aggregation is numeric for a non-numeric column
        numeric_agg_funcs = ['mean', 'median', 'std', 'var', 'sum']
        if agg_func in numeric_agg_funcs and not pd.api.types.is_numeric_dtype(df[agg_col]):
            error_message = f"Error: Aggregation function '{agg_func}' can only be applied to numeric columns. Column '{agg_col}' is not numeric."
            return None, error_message
        
        if isinstance(agg_func, str):
            aggregated_df = df.groupby(group_by_cols)[agg_col].agg(agg_func).reset_index()
        elif isinstance(agg_func, list):
            aggregated_df = df.groupby(group_by_cols)[agg_col].agg(agg_func).reset_index()
            aggregated_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in aggregated_df.columns.values]
        else:
            return None, "Error: agg_func must be a string or a list of strings."
        
        return aggregated_df, None
    except Exception as e:
        error_message = f"An error occurred during data aggregation: {e}"
        return None, error_message
