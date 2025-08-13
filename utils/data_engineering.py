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
            # Handle division by zero
            if (df_engineered[col2] == 0).any():
                return None, "Error: Division by zero is not allowed."
            df_engineered[new_col_name] = df_engineered[col1] / df_engineered[col2]
        else:
            return None, "Error: Invalid operation specified. Use 'add', 'subtract', 'multiply', or 'divide'."
        
        return df_engineered, None
    except Exception as e:
        return None, f"An error occurred during feature creation: {e}"

def apply_encoding(df, columns, encoding_type, **kwargs):
    """
    Applies various encoding techniques to specified categorical columns.

    Technical: This function provides a unified interface for several encoding
    methods. Label Encoding is for binary data. Ordinal Encoding is for
    ordered categorical data. One-Hot Encoding creates new binary columns
    for each category. Frequency Encoding replaces categories with their
    frequency in the dataset.

    Layman: This is where we turn text-based categories (like 'car' or 'radio/TV')
    into numbers that a computer can understand. We have different ways to do this,
    depending on whether the categories have a natural order or not.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to encode.
        encoding_type (str): The type of encoding to apply ('one_hot', 'label', 'ordinal', 'frequency').
        **kwargs: Additional arguments for specific encoders (e.g., categories for OrdinalEncoder).

    Returns:
        tuple: The DataFrame with encoded columns and an error message (if any).
    """
    df_encoded = df.copy()
    error_message = None
    try:
        for col in columns:
            if not pd.api.types.is_categorical_dtype(df_encoded[col]) and not pd.api.types.is_object_dtype(df_encoded[col]):
                continue

            if encoding_type == 'label':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
            elif encoding_type == 'ordinal':
                oe = OrdinalEncoder(**kwargs)
                df_encoded[col] = oe.fit_transform(df_encoded[[col]])
            elif encoding_type == 'one_hot':
                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded_cols = ohe.fit_transform(df_encoded[[col]])
                encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out([col]), index=df_encoded.index)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded_df], axis=1)
            elif encoding_type == 'frequency':
                freq_map = df_encoded[col].value_counts(normalize=True)
                df_encoded[col] = df_encoded[col].map(freq_map)
            else:
                error_message = f"Error: Invalid encoding type '{encoding_type}'."
                return df, error_message
        return df_encoded, None
    except Exception as e:
        error_message = f"An error occurred during encoding: {e}"
        return df, error_message
