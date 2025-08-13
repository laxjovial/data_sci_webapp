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

    try:
        # Attempt to convert value to the column's type
        col_type = df_filtered[column].dtype
        converted_value = pd.Series([value]).astype(col_type).iloc[0]

        if operator == '>':
            df_filtered = df_filtered[df_filtered[column] > converted_value]
        elif operator == '<':
            df_filtered = df_filtered[df_filtered[column] < converted_value]
        elif operator == '==':
            df_filtered = df_filtered[df_filtered[column] == converted_value]
        elif operator == '!=':
            df_filtered = df_filtered[df_filtered[column] != converted_value]
        elif operator == '>=':
            df_filtered = df_filtered[df_filtered[column] >= converted_value]
        elif operator == '<=':
            df_filtered = df_filtered[df_filtered[column] <= converted_value]
        elif operator == 'contains':
            if pd.api.types.is_string_dtype(col_type):
                 df_filtered = df_filtered[df_filtered[column].str.contains(str(converted_value), case=False, na=False)]
            else:
                error_message = "Error: 'contains' operator only works on text columns."
        else:
            error_message = "Error: Invalid operator specified."

    except Exception as e:
        error_message = f"Error applying filter: {e}"

    if error_message:
        return df, error_message

    return df_filtered, None
