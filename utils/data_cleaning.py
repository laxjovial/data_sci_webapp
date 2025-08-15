# utils/data_cleaning.py

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

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

def handle_outliers_iqr(df, columns, strategy='remove', multiplier=1.5):
    """
    Handles outliers in numerical columns using the Interquartile Range (IQR) method.

    Technical: Calculates the IQR (Q3 - Q1) for each specified column. Outliers are
    identified as data points that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    The function can either remove these rows or cap the values at the outlier boundaries.

    Layman: This function finds and deals with extreme values (outliers) in your data.
    You can choose to either completely remove the rows containing these outliers or
    replace the outlier values with the nearest "normal" value.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of numerical column names to process.
        strategy (str): The strategy to handle outliers ('remove' or 'cap').
        multiplier (float): The IQR multiplier to determine outlier boundaries.

    Returns:
        tuple: A tuple containing the new DataFrame and an error message (if any).
    """
    df_cleaned = df.copy()
    for col in columns:
        if col not in df_cleaned.columns or not pd.api.types.is_numeric_dtype(df_cleaned[col]):
            return None, f"Error: Column '{col}' is not a valid numerical column."

        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        if strategy == 'remove':
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        elif strategy == 'cap':
            df_cleaned[col] = np.where(df_cleaned[col] < lower_bound, lower_bound, df_cleaned[col])
            df_cleaned[col] = np.where(df_cleaned[col] > upper_bound, upper_bound, df_cleaned[col])
        else:
            return None, "Error: Invalid strategy. Choose 'remove' or 'cap'."

    return df_cleaned, None

def apply_regex_cleaning(df, column, pattern, replacement):
    """
    Cleans a text column using a regular expression.

    Technical: Uses pandas' `str.replace()` method with `regex=True`. This allows for
    finding and replacing substrings in a column based on a specified regex pattern.

    Layman: This is an advanced "find and replace" for your text data. You can use it
    to clean up complex text patterns, like removing special characters or extracting
    specific parts of the text.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the text column to clean.
        pattern (str): The regex pattern to search for.
        replacement (str): The string to replace the matched pattern with.

    Returns:
        tuple: A tuple containing the new DataFrame and an error message (if any).
    """
    df_cleaned = df.copy()
    if column not in df_cleaned.columns or not pd.api.types.is_string_dtype(df_cleaned[column]):
        return None, f"Error: Column '{column}' is not a valid text column."

    try:
        df_cleaned[column] = df_cleaned[column].str.replace(pattern, replacement, regex=True)
    except Exception as e:
        return None, f"Error applying regex: {e}"

    return df_cleaned, None

def impute_knn(df, columns, n_neighbors=5):
    """
    Imputes missing values using K-Nearest Neighbors.

    Technical: Utilizes scikit-learn's `KNNImputer`. This method models each feature
    with missing values as a function of other features, and imputes missing values
    based on the values of the k-nearest neighbors in the dataset.

    Layman: This is a smart way to fill in missing values. Instead of just using the
    average, it looks at similar data points (neighbors) to make an educated guess
    about what the missing value should be.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of numerical column names to impute.
        n_neighbors (int): The number of neighbors to use for imputation.

    Returns:
        tuple: A tuple containing the new DataFrame and an error message (if any).
    """
    df_imputed = df.copy()

    numerical_cols = df_imputed[columns].select_dtypes(include=np.number).columns.tolist()
    if len(numerical_cols) != len(columns):
        return None, "Error: All columns for KNN imputation must be numerical."

    try:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        # The imputer returns a numpy array, so we need to put it back into a DataFrame
        df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
    except Exception as e:
        return None, f"Error during KNN imputation: {e}"

    return df_imputed, None
