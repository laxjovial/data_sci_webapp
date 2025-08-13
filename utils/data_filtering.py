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
        tuple: A tuple containing the filtered DataFrame and an error message (if any).
    """
    if column not in df.columns:
        return None, f"Error: Column '{column}' not found."
    
    try:
        # Try to convert the value to the column's data type
        col_type = df[column].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            value = float(value)
        elif pd.api.types.is_datetime64_any_dtype(col_type):
            value = pd.to_datetime(value)
        else:
            value = str(value)
            
        if operator == '>':
            filtered_df = df[df[column] > value]
        elif operator == '<':
            filtered_df = df[df[column] < value]
        elif operator == '==':
            filtered_df = df[df[column] == value]
        elif operator == '!=':
            filtered_df = df[df[column] != value]
        elif operator == '>=':
            filtered_df = df[df[column] >= value]
        elif operator == '<=':
            filtered_df = df[df[column] <= value]
        else:
            return None, f"Error: Invalid operator '{operator}'."
            
        return filtered_df, None
    except (ValueError, TypeError) as e:
        return None, f"Error: Failed to filter. Check if the value type matches the column type. {e}"
