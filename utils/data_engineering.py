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

def apply_encoding(df, columns, encoding_type):
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

    Returns:
        tuple: The DataFrame with encoded columns and an error message (if any).
    """
    df_encoded = df.copy()
    error_message = None

    if encoding_type == 'label':
        # Label Encoding is best for binary categories like 'male'/'female'
        for col in columns:
            if df_encoded[col].nunique() <= 2:
                try:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                except Exception as e:
                    error_message = f"Error applying Label Encoding to '{col}': {e}"
                    return df, error_message
            else:
                error_message = f"Label Encoding is recommended for binary columns. '{col}' has more than 2 unique values."
                return df, error_message
                
    elif encoding_type == 'ordinal':
        # Ordinal Encoding is for categories with a clear order, like 'little' < 'moderate' < 'rich'
        try:
            oe = OrdinalEncoder()
            df_encoded[columns] = oe.fit_transform(df_encoded[columns])
        except Exception as e:
            error_message = f"Error applying Ordinal Encoding: {e}"
            return df, error_message
            
    elif encoding_type == 'one_hot':
        # One-Hot Encoding is for nominal categories with no order.
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = ohe.fit_transform(df_encoded[columns])
            encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(columns))
            
            df_encoded.drop(columns=columns, inplace=True)
            df_encoded.reset_index(drop=True, inplace=True)
            encoded_df.reset_index(drop=True, inplace=True)
            
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        except Exception as e:
            error_message = f"Error applying One-Hot Encoding: {e}"
            return df, error_message
            
    elif encoding_type == 'frequency':
        # Frequency Encoding replaces categories with the proportion of their occurrence.
        try:
            for col in columns:
                frequency_map = df_encoded[col].value_counts(normalize=True)
                df_encoded[col] = df_encoded[col].map(frequency_map)
        except Exception as e:
            error_message = f"Error applying Frequency Encoding: {e}"
            return df, error_message

    else:
        error_message = "Invalid encoding type selected."

    return df_encoded, error_message


def scale_features(df, columns, scaler_type):
    """
    Scales numerical features using a specified method.

    Technical: This function uses either StandardScaler or MinMaxScaler from
    scikit-learn to transform numerical data. StandardScaler normalizes the data
    to have a mean of 0 and a standard deviation of 1.

    Layman: This is like putting all your numbers on the same playing field. If you
    have a column for "age" (e.g., 20-80) and a column for "income" (e.g., $10k-$1M),
    a model might think income is more important just because the numbers are
    bigger. Scaling evens out the numbers so the model can focus on the patterns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of numerical column names to scale.
        scaler_type (str): The type of scaler to use ('standard', 'minmax').

    Returns:
        tuple: A tuple containing the new DataFrame and an error message (if any).
    """
    df_engineered = df.copy()
    try:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            return None, "Error: Invalid scaler type. Use 'standard' or 'minmax'."

        df_engineered[columns] = scaler.fit_transform(df_engineered[columns])
        return df_engineered, None
    except Exception as e:
        return None, f"An error occurred during feature scaling: {e}"

def rename_and_drop_columns(df, new_column_name=None, old_column_name=None, columns_to_drop=None):
    """
    Renames a column and/or drops multiple columns from the DataFrame.

    Technical: This is a utility function that uses the pandas.DataFrame.rename
    and pandas.DataFrame.drop methods to modify the DataFrame's column structure.
    It's designed to be a flexible way to handle both renaming and dropping in one
    function.

    Layman: This is for organizing your data. You can give a column a new, more
    descriptive name (like changing 'col1' to 'product_id'), and you can get rid of
    columns that you don't need for your analysis.

    Args:
        df (pd.DataFrame): The input DataFrame.
        new_column_name (str, optional): The new name for a column.
        old_column_name (str, optional): The old name of the column to be renamed.
        columns_to_drop (list, optional): A list of columns to be dropped.

    Returns:
        pd.DataFrame: The DataFrame with renamed and/or dropped columns.
    """
    df_engineered = df.copy()
    if old_column_name and new_column_name:
        df_engineered.rename(columns={old_column_name: new_column_name}, inplace=True)
    if columns_to_drop:
        df_engineered.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
    return df_engineered, None
