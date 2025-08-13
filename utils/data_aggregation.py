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
        # Check if aggregation is numeric for a non-numeric column
        if agg_func in ['mean', 'sum', 'median', 'std', 'var'] and not pd.api.types.is_numeric_dtype(df[agg_col]):
            error_message = f"Error: Aggregation function '{agg_func}' can only be applied to numeric columns."
            return None, error_message

        aggregated_df = df.groupby(group_by_cols)[agg_col].agg(agg_func).reset_index()
        return aggregated_df, None
    except Exception as e:
        error_message = f"An error occurred during aggregation: {e}"
        return None, error_message

def pivot_table(df, index_cols, column_cols, value_col, agg_func):
    """
    Creates a pivot table from the DataFrame.

    Technical: Uses pandas' pivot_table() function, which reshapes the data
    based on specified index, column, and value columns, applying an aggregation
    function to the values.

    Layman: A pivot table is a powerful way to summarize and reorganize your
    data. For instance, you could see 'Sales' (values) for each 'Product' (rows)
    across different 'Regions' (columns).

    Args:
        df (pd.DataFrame): The input DataFrame.
        index_cols (list): Columns to use for the pivot table index (rows).
        column_cols (list): Columns to use for the pivot table columns.
        value_col (str): The column to aggregate.
        agg_func (str): The aggregation function.

    Returns:
        pd.DataFrame: The resulting pivot table.
        str: An error message, if any.
    """
    error_message = None
    try:
        pivot_df = df.pivot_table(values=value_col, index=index_cols, columns=column_cols, aggfunc=agg_func)
        # We reset index to make it a more standard DataFrame for display
        return pivot_df.reset_index(), None
    except Exception as e:
        error_message = f"An error occurred creating the pivot table: {e}"
        return None, error_message
