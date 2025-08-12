# utils/data_engineering.py
import pandas as pd

def create_new_feature(df, col1, col2, new_col_name, operation):
    """
    Creates a new feature by combining two existing columns with a mathematical operation.

    Technical: This function takes two column names, a new name for the resulting
    column, and a specified operation ('add', 'subtract', 'multiply', 'divide').
    It performs the operation on the specified columns and adds the result as
    a new column in the DataFrame. It includes error handling for invalid operations
    or non-numeric data.

    Layman: This lets you create a new, smart column. For example, if you have
    a 'Length' and a 'Width' column, you can multiply them together to create
    a new 'Area' column. This helps find new patterns in your data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The name of the first column.
        col2 (str): The name of the second column.
        new_col_name (str): The name of the new column to be created.
        operation (str): The mathematical operation to perform ('add', 'subtract', 'multiply', 'divide').

    Returns:
        tuple: (pd.DataFrame, str) The modified DataFrame and an error message (if any).
    """
    df_engineered = df.copy()
    try:
        if operation == 'add':
            df_engineered[new_col_name] = df_engineered[col1] + df_engineered[col2]
        elif operation == 'subtract':
            df_engineered[new_col_name] = df_engineered[col1] - df_engineered[col2]
        elif operation == 'multiply':
            df_engineered[new_col_name] = df_engineered[col1] * df_engineered[col2]
        elif operation == 'divide':
            # Handle division by zero
            df_engineered[new_col_name] = df_engineered[col1] / df_engineered[col2].replace(0, 1)
        else:
            return df, "Invalid operation specified. Please choose from 'add', 'subtract', 'multiply', 'divide'."
        return df_engineered, None
    except KeyError:
        return df, "One or both of the specified columns were not found."
    except TypeError:
        return df, "The selected columns must contain numeric data to perform this operation."
    except Exception as e:
        return df, f"An unexpected error occurred: {e}"


def one_hot_encode_column(df, column):
    """
    Performs one-hot encoding on a specified categorical column.

    Technical: This function uses pandas' get_dummies() to convert a categorical
    column into a series of binary columns. This is a standard practice for
    preparing non-numeric data for machine learning models.

    Layman: If you have a column with categories like 'Red', 'Green', and 'Blue',
    this creates new columns for each category (e.g., 'Color_Red', 'Color_Green')
    and puts a '1' in the correct column for each row. This allows a computer to
    understand the categories.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the categorical column to encode.

    Returns:
        tuple: (pd.DataFrame, str) The modified DataFrame and an error message (if any).
    """
    df_engineered = df.copy()
    try:
        if column not in df_engineered.columns:
            return df, f"Column '{column}' not found in the DataFrame."
        
        # Check if the column is a categorical type
        if pd.api.types.is_numeric_dtype(df_engineered[column]):
            return df, f"Column '{column}' is numeric and cannot be one-hot encoded."

        dummies = pd.get_dummies(df_engineered[column], prefix=column, drop_first=True)
        df_engineered = pd.concat([df_engineered, dummies], axis=1)
        df_engineered = df_engineered.drop(column, axis=1)

        return df_engineered, None
    except Exception as e:
        return df, f"An unexpected error occurred: {e}"
