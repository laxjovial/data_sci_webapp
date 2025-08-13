import pandas as pd

def filter_dataframe(df, column, operator, value):
    """
    Filters the DataFrame based on a condition.

    Technical: This function dynamically builds a filter query. It first tries
    to convert the 'value' to the same data type as the DataFrame column to
    ensure type-safe comparison. It then uses boolean indexing to filter
    the DataFrame based on the specified operator.

    Layman: This is like a search filter for your data. You can pick a column
    (like 'Age'), an operator (like 'greater than'), and a value (like '30')
    to see only the rows that match your criteria.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to filter on.
        operator (str): The comparison operator (e.g., '>', '<', '==', '!=', '>=', '<=').
        value (str): The value to compare against.

    Returns:
        pd.DataFrame: The filtered DataFrame.
        str: An error message, if any.
    """
    df_filtered = df.copy()
    error_message = None

    if column not in df_filtered.columns:
        return df, f"Error: Column '{column}' not found."

    supported_operators = ['>', '<', '==', '!=', '>=', '<=']
    if operator not in supported_operators:
        return df, f"Error: Invalid operator '{operator}'. Supported operators are {supported_operators}."

    try:
        # Attempt to convert value to the column's data type
        col_dtype = df_filtered[column].dtype
        if pd.api.types.is_numeric_dtype(col_dtype):
            typed_value = float(value)
        else:
            typed_value = value

        # Apply the filter
        if operator == '>':
            df_filtered = df_filtered[df_filtered[column] > typed_value]
        elif operator == '<':
            df_filtered = df_filtered[df_filtered[column] < typed_value]
        elif operator == '==':
            df_filtered = df_filtered[df_filtered[column] == typed_value]
        elif operator == '!=':
            df_filtered = df_filtered[df_filtered[column] != typed_value]
        elif operator == '>=':
            df_filtered = df_filtered[df_filtered[column] >= typed_value]
        elif operator == '<=':
            df_filtered = df_filtered[df_filtered[column] <= typed_value]

        return df_filtered, None
    except (ValueError, TypeError) as e:
        error_message = f"Error: Could not convert filter value '{value}' to match column '{column}' type. {e}"
        return df, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred during filtering: {e}"
        return df, error_message
