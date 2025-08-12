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

def remove_duplicates(df):
    """
    Removes duplicate rows from the DataFrame.

    Technical: Uses the pandas.DataFrame.drop_duplicates method. By default, it
    removes all rows that are exact duplicates, keeping only the first occurrence.

    Layman: This will find and remove any rows in your data that are identical
    to another row, so you don't have the same information twice.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
    """
    return df.drop_duplicates()

def standardize_text(df, columns):
    """
    Standardizes text data in specified columns.

    Technical: Iterates through the specified columns and applies string methods
    to trim whitespace, convert to lowercase, and handle null values gracefully.

    Layman: This makes all the text in a column look the same. For example, it
    will change " NEW YORK " and "New York" to "new york", so the app sees them
    as the same thing.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to standardize.

    Returns:
        pd.DataFrame: The DataFrame with text data standardized.
    """
    df_cleaned = df.copy()
    for col in columns:
        if pd.api.types.is_string_dtype(df_cleaned[col]):
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()
    return df_cleaned

def handle_outliers(df, column, method, value=None):
    """
    Handles outliers in a numerical column using different methods.

    Technical: Implements the IQR (Interquartile Range) method to detect outliers
    and allows for either removal or capping. The IQR method calculates the
    upper and lower bounds (Q3 + 1.5*IQR and Q1 - 1.5*IQR) and either filters
    out values outside this range or replaces them with the boundary values.

    Layman: This is for dealing with numbers that are way too big or too small
    compared to the rest of the data. You can either get rid of those extreme
    numbers or "cap" them, which means replacing them with the highest or lowest
    acceptable value. This helps prevent them from messing up your analysis.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The numerical column to check for outliers.
        method (str): The method to use ('remove' or 'cap').
        value (float, optional): A specific value to use for capping.

    Returns:
        pd.DataFrame: The DataFrame with outliers handled.
        str: An error message if the column is not numeric.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df, "Error: Outlier handling is only for numeric columns."

    df_cleaned = df.copy()
    Q1 = df_cleaned[column].quantile(0.25)
    Q3 = df_cleaned[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    if method == 'remove':
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
    elif method == 'cap':
        df_cleaned[column] = np.where(df_cleaned[column] > upper_bound, upper_bound, df_cleaned[column])
        df_cleaned[column] = np.where(df_cleaned[column] < lower_bound, lower_bound, df_cleaned[column])
    elif method == 'custom_cap' and value is not None:
        df_cleaned[column] = np.where(df_cleaned[column] > value, value, df_cleaned[column])
        df_cleaned[column] = np.where(df_cleaned[column] < -value, -value, df_cleaned[column])
        
    return df_cleaned, None

def correct_inconsistencies(df, column, mapping_dict):
    """
    Corrects inconsistent entries in a categorical column using a mapping dictionary.

    Technical: Uses the pandas.DataFrame.replace method to map old, inconsistent
    values to new, standardized values as defined by a dictionary.

    Layman: This helps you fix typos and different spellings in a column. For
    example, you can tell the app to change "NY" and "New York City" to
    the standard "New York."

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column containing inconsistent values.
        mapping_dict (dict): A dictionary with inconsistent values as keys
                             and the correct value as the value.

    Returns:
        pd.DataFrame: The DataFrame with corrected entries.
    """
    df_cleaned = df.copy()
    if column in df_cleaned.columns:
        df_cleaned[column] = df_cleaned[column].replace(mapping_dict)
    return df_cleaned
