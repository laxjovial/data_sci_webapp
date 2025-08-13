# utils/data_engineering.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
import numpy as np

def create_new_feature(df, col1, col2, operation, new_col_name):
    """
    Creates a new feature by performing a mathematical operation on two existing columns.

    Technical: This function takes a DataFrame and two numeric column names,
    applies a specified operation (+, -, *, /), and adds the result as a new
    column to the DataFrame. It includes error handling for non-numeric columns
    and division by zero.

    Layman: This is like a calculator for your data. You can take two columns,
    like "height" and "weight," and create a new column, like "BMI," by
    dividing them. This helps create more meaningful data for a model to learn from.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The name of the first column.
        col2 (str): The name of the second column.
        operation (str): The mathematical operation to perform ('add', 'subtract', 'multiply', 'divide').
        new_col_name (str): The name for the new column.

    Returns:
        tuple: A tuple containing the new DataFrame and an error message (if any).
    """
    df_engineered = df.copy()
    if col1 not in df_engineered.columns or col2 not in df_engineered.columns:
        return None, "Error: One or both of the specified columns do not exist."
    
    if not pd.api.types.is_numeric_dtype(df_engineered[col1]) or not pd.api.types.is_numeric_dtype(df_engineered[col2]):
        return None, "Error: Selected columns must be numeric for this operation."

    try:
        if operation == 'add':
            df_engineered[new_col_name] = df_engineered[col1] + df_engineered[col2]
        elif operation == 'subtract':
            df_engineered[new_col_name] = df_engineered[col1] - df_engineered[col2]
        elif operation == 'multiply':
            df_engineered[new_col_name] = df_engineered[col1] * df_engineered[col2]
        elif operation == 'divide':
            if (df_engineered[col2] == 0).any():
                return None, "Error: Division by zero is not allowed."
            df_engineered[new_col_name] = df_engineered[col1] / df_engineered[col2]
        else:
            return None, "Error: Invalid operation specified. Use 'add', 'subtract', 'multiply', or 'divide'."
        return df_engineered, None
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

def apply_encoding(df, column, encoding_type):
    """
    Applies encoding to a categorical column.

    Technical: This function uses popular scikit-learn encoders.
    One-Hot Encoding creates new binary columns for each category.
    Label Encoding converts each category into a unique integer.

    Layman: This turns text categories, like 'Red', 'Green', 'Blue', into
    numbers that a machine learning model can understand. This is a crucial
    step for preparing categorical data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to encode.
        encoding_type (str): The type of encoding to apply ('one-hot', 'label', 'ordinal').

    Returns:
        tuple: A tuple containing the new DataFrame and an error message (if any).
    """
    df_encoded = df.copy()
    if column not in df_encoded.columns:
        return None, f"Error: Column '{column}' not found."
    
    try:
        if encoding_type == 'one-hot':
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = encoder.fit_transform(df_encoded[[column]])
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column]))
            df_encoded = pd.concat([df_encoded.drop(column, axis=1), encoded_df], axis=1)
        elif encoding_type == 'label':
            encoder = LabelEncoder()
            df_encoded[f'{column}_encoded'] = encoder.fit_transform(df_encoded[column])
        elif encoding_type == 'ordinal':
            encoder = OrdinalEncoder()
            df_encoded[f'{column}_ordinal'] = encoder.fit_transform(df_encoded[[column]])
        else:
            return None, "Error: Invalid encoding type. Use 'one-hot', 'label', or 'ordinal'."
        
        return df_encoded, None
    except Exception as e:
        return None, f"An unexpected error occurred during encoding: {e}"

def bin_column(df, column, bins):
    """
    Divides a numeric column into discrete bins.

    Technical: This function uses pandas.cut() to segment and sort data values
    into bins. This process, also known as 'binning' or 'discretization',
    converts continuous numeric data into categorical data.

    Layman: This is useful for turning a column with a wide range of numbers
    (like 'Age') into a column with a few groups (like 'Child', 'Teen', 'Adult').
    This can make data easier to work with or visualize.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to bin.
        bins (int): The number of bins to create.

    Returns:
        tuple: A tuple containing the new DataFrame and an error message (if any).
    """
    df_binned = df.copy()
    if column not in df_binned.columns:
        return None, f"Error: Column '{column}' not found."
    if not pd.api.types.is_numeric_dtype(df_binned[column]):
        return None, "Error: Column must be numeric for binning."
    
    try:
        df_binned[f'{column}_binned'] = pd.cut(df_binned[column], bins=bins, include_lowest=True, right=False)
        return df_binned, None
    except Exception as e:
        return None, f"An unexpected error occurred during binning: {e}"
