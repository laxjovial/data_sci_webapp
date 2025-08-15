import dask.dataframe as dd
import pandas as pd
import io

def filter_dataframe(df, column, operator, value, value2=''):
    """
    Filters a Dask DataFrame based on a condition, with support for a variety of operators.

    Technical: This function leverages Dask's Pandas-like API to perform filtering. It
    handles different data types (numerical, datetime, string) and supports advanced
    operators like 'contains', 'isin', and 'between'. Dask's lazy evaluation ensures
    that the filtering is performed efficiently across partitions.

    Layman: This is a powerful search filter for your data. You can now search for
    text that "contains" a certain word, or filter for rows where a category is "in" a
    list of values you provide. You can also filter for numbers or dates "between" two points.

    Args:
        df (dd.DataFrame): The input Dask DataFrame.
        column (str): The column to filter on.
        operator (str): The comparison operator (e.g., '>', '<', '==', 'contains', 'isin', 'between').
        value (str): The primary value to compare against.
        value2 (str, optional): The second value for operators like 'between'.

    Returns:
        tuple: A tuple containing the filtered Dask DataFrame and an error message (if any).
    """
    if column not in df.columns:
        return None, f"Error: Column '{column}' not found."

    df_filtered = df.copy()
    col_dtype = df[column].dtype

    try:
        # --- Handle operators for numerical and datetime columns ---
        if pd.api.types.is_numeric_dtype(col_dtype) or pd.api.types.is_datetime64_any_dtype(col_dtype):
            # Dask can't reliably convert types on a Series, so we use pandas for type casting
            try:
                value = pd.to_numeric(value) if pd.api.types.is_numeric_dtype(col_dtype) else pd.to_datetime(value)
            except (ValueError, TypeError):
                return None, "Error: Invalid filter value provided for the column's type."
            
            if operator == '>':
                df_filtered = df_filtered[df_filtered[column] > value]
            elif operator == '<':
                df_filtered = df_filtered[df_filtered[column] < value]
            elif operator == '==':
                df_filtered = df_filtered[df_filtered[column] == value]
            elif operator == '!=':
                df_filtered = df_filtered[df_filtered[column] != value]
            elif operator == '>=':
                df_filtered = df_filtered[df_filtered[column] >= value]
            elif operator == '<=':
                df_filtered = df_filtered[df_filtered[column] <= value]
            elif operator == 'between':
                try:
                    value2 = pd.to_numeric(value2) if pd.api.types.is_numeric_dtype(col_dtype) else pd.to_datetime(value2)
                except (ValueError, TypeError):
                    return None, "Error: Invalid second filter value for 'between' operator."
                df_filtered = df_filtered[df_filtered[column].between(value, value2)]
            else:
                return None, f"Error: Operator '{operator}' is not valid for numerical/date columns."

        # --- Handle operators for string columns ---
        elif pd.api.types.is_string_dtype(col_dtype) or col_dtype == 'object':
            value = str(value)
            if operator == '==':
                df_filtered = df_filtered[df_filtered[column] == value]
            elif operator == '!=':
                df_filtered = df_filtered[df_filtered[column] != value]
            elif operator == 'contains':
                df_filtered = df_filtered[df_filtered[column].str.contains(value, na=False)]
            elif operator == 'startswith':
                df_filtered = df_filtered[df_filtered[column].str.startswith(value, na=False)]
            elif operator == 'endswith':
                df_filtered = df_filtered[df_filtered[column].str.endswith(value, na=False)]
            elif operator in ['isin', 'notin']:
                # Create a list from a comma-separated string
                filter_list = [item.strip() for item in value.split(',')]
                if operator == 'isin':
                    df_filtered = df_filtered[df_filtered[column].isin(filter_list)]
                else:  # 'notin'
                    df_filtered = df_filtered[~df_filtered[column].isin(filter_list)]
            else:
                return None, f"Error: Operator '{operator}' is not valid for text columns."
        else:
            return None, "Error: Filtering is only supported for numerical, datetime, and text columns."
            
        return df_filtered, None
    except Exception as e:
        return None, f"Error during filtering: {e}. Please check your input values and operator."