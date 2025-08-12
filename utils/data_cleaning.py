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
        pd.DataFrame: The DataFrame with missing values handled.
    """
    df_cleaned = df.copy()
    if strategy == 'drop':
        df_cleaned.dropna(subset=columns, inplace=True)
    else:
        for col in columns:
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            elif strategy == 'mode':
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    return df_cleaned

def rename_column(df, old_col, new_col):
    """
    Renames a single column in the DataFrame.

    Technical: Uses the pandas.DataFrame.rename method with a dictionary mapping.
    This is an in-place operation that changes the column name permanently.

    Layman: This simply changes the name of a column to whatever you want.

    Args:
        df (pd.DataFrame): The input DataFrame.
        old_col (str): The current name of the column.
        new_col (str): The new name for the column.

    Returns:
        pd.DataFrame: The DataFrame with the column renamed.
    """
    if old_col in df.columns and new_col:
        return df.rename(columns={old_col: new_col})
    return df

def convert_dtype(df, column, new_type):
    """
    Converts a column's data type.

    Technical: Utilizes pandas' astype() method. For datetime conversion, it
    calls pd.to_datetime() with errors='coerce' to handle parsing issues,
    converting invalid dates to 'NaT' (Not a Time).

    Layman: This changes the type of information in a column, for example,
    from text to a number, or from a general text format to a specific date format.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to convert.
        new_type (str): The target data type ('int', 'float', 'str', 'datetime').

    Returns:
        pd.DataFrame: The DataFrame with the column type converted.
        str: An error message if conversion fails, otherwise None.
    """
    df_cleaned = df.copy()
    error_message = None
    try:
        if new_type == 'datetime':
            df_cleaned[column] = pd.to_datetime(df_cleaned[column], errors='coerce')
        else:
            df_cleaned[column] = df_cleaned[column].astype(new_type)
    except Exception as e:
        error_message = f"Error converting column '{column}' to type '{new_type}': {e}"
    return df_cleaned, error_message
