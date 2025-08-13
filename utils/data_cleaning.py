# utils/data_cleaning.py

import pandas as pd
import numpy as np

def handle_missing_values(df, columns, strategy):
    """
    Handles missing values in specified columns based on a given strategy.

    Technical: Utilizes pandas' fillna() and dropna() methods. For imputation, it
    calculates the mean, median, or mode of the specified columns and fills the
    NaN values. For dropping, it removes rows where any of the specified columns
    have a missing value.

    Layman: This function is for fixing missing data. You can either fill in the
    blanks with an average value (mean/median), the most common value (mode),
    or simply remove the rows that have any missing information.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to process.
        strategy (str): The imputation strategy ('mean', 'median', 'mode', 'drop').

    Returns:
        tuple: A tuple containing the new DataFrame and an error message (if any).
    """
    df_cleaned = df.copy()
    if not all(col in df_cleaned.columns for col in columns):
        return None, "Error: One or more specified columns do not exist."
    
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna(subset=columns)
    else:
        for col in columns:
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                if strategy == 'mean':
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                elif strategy == 'median':
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            elif strategy == 'mode':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            else:
                return None, f"Error: Invalid strategy '{strategy}' for column '{col}'."
                
    return df_cleaned, None

def rename_column(df, old_col, new_col):
    """
    Renames a single column in the DataFrame.

    Technical: Uses the pandas.DataFrame.rename method with a dictionary mapping.
    This returns a new DataFrame with the column renamed.

    Layman: This simply changes the name of a column in your dataset.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        old_col (str): The current name of the column.
        new_col (str): The new name for the column.

    Returns:
        tuple: A tuple containing the new DataFrame and an error message (if any).
    """
    df_renamed = df.copy()
    if old_col not in df_renamed.columns:
        return None, f"Error: Column '{old_col}' not found."
    df_renamed = df_renamed.rename(columns={old_col: new_col})
    return df_renamed, None

def convert_dtype(df, column, new_type):
    """
    Converts a column to a new data type.

    Technical: Utilizes pandas' astype() method to cast a column to a different
    data type. It includes error handling to manage cases where the conversion
    is not possible.

    Layman: This changes the type of data in a column, for example, from text
    to a number. This is important when you want to perform calculations
    on a column that is currently saved as text.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to convert.
        new_type (str): The new data type ('int', 'float', 'str', 'datetime').

    Returns:
        tuple: A tuple containing the new DataFrame and an error message (if any).
    """
    df_converted = df.copy()
    if column not in df_converted.columns:
        return None, f"Error: Column '{column}' not found."
    
    try:
        if new_type == 'datetime':
            df_converted[column] = pd.to_datetime(df_converted[column], errors='coerce')
            if df_converted[column].isnull().any():
                return None, "Error: Could not convert all values to datetime."
        else:
            df_converted[column] = df_converted[column].astype(new_type)
    except (ValueError, TypeError) as e:
        return None, f"Error converting column '{column}' to type '{new_type}': {e}"
        
    return df_converted, None

def remove_duplicates(df):
    """
    Removes duplicate rows from the DataFrame.

    Technical: Uses the pandas.DataFrame.drop_duplicates() method. This operation
    identifies and removes rows that have identical values across all columns.

    Layman: This function helps clean up your data by deleting any exact copies
    of rows, ensuring that each piece of information is unique.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing the new DataFrame and an error message (if any).
    """
    df_cleaned = df.copy()
    df_cleaned.drop_duplicates(inplace=True)
    return df_cleaned, None

def standardize_text(df, columns):
    """
    Standardizes text in specified columns by converting to lowercase,
    stripping whitespace, and handling common patterns.
    
    Technical: This function applies a series of string manipulation methods
    (str.lower(), str.strip()) to the specified columns. It's a key part
    of preparing text data for natural language processing (NLP) or
    simple comparison.

    Layman: This function cleans up text. It makes sure that all the text
    in a column is consistent, for example, by making everything lowercase
    and removing extra spaces. This prevents errors when you're trying to
    group or count text data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names containing text data.
        
    Returns:
        tuple: A tuple containing the new DataFrame and an error message (if any).
    """
    df_cleaned = df.copy()
    for col in columns:
        if col not in df_cleaned.columns:
            return None, f"Error: Column '{col}' not found."
        if pd.api.types.is_string_dtype(df_cleaned[col]):
            df_cleaned[col] = df_cleaned[col].str.lower().str.strip()
    return df_cleaned, None
