import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def handle_missing_values(df, columns, strategy):
    """Handles missing values in specified columns of a Pandas DataFrame."""
    df_cleaned = df.copy()
    if not all(col in df_cleaned.columns for col in columns):
        return None, "Error: One or more specified columns do not exist."
    
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna(subset=columns)
    else:
        for col in columns:
            if strategy == 'mean':
                fill_value = df_cleaned[col].mean()
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
            elif strategy == 'median':
                fill_value = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
            elif strategy == 'mode':
                fill_value = df_cleaned[col].mode()[0]
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    return df_cleaned, None

def rename_column(df, old_col, new_col):
    """Renames a column in a Pandas DataFrame."""
    return df.rename(columns={old_col: new_col}), None

def convert_dtype(df, column, new_type):
    """Converts a Pandas DataFrame column to a new data type."""
    try:
        df[column] = df[column].astype(new_type)
        return df, None
    except Exception as e:
        return None, f"Error converting dtype: {e}"

def remove_duplicates(df, subset=None):
    """Removes duplicate rows from a Pandas DataFrame."""
    return df.drop_duplicates(subset=subset), None

def standardize_text(df, columns):
    """Standardizes text columns in a Pandas DataFrame."""
    df_cleaned = df.copy()
    for col in columns:
        if col not in df_cleaned.columns:
            return None, f"Error: Column '{col}' not found."
        # Ensure the column is of string type before using .str accessor
        if pd.api.types.is_string_dtype(df_cleaned[col]):
            df_cleaned[col] = df_cleaned[col].str.lower().str.strip()
    return df_cleaned, None

def handle_outliers_iqr(df, columns, strategy='remove', multiplier=1.5):
    """
    Handles outliers in numerical columns using the Interquartile Range (IQR) method on a Pandas DataFrame.

    Technical: Calculates the IQR (Q3 - Q1) for each specified column. Outliers are
    identified as data points that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    The function can either remove these rows or cap the values at the outlier boundaries.

    Layman: This function finds and deals with extreme values (outliers) in your data.
    You can choose to either completely remove the rows containing these outliers or
    replace the outlier values with the nearest "normal" value.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        columns (list): A list of numerical column names to process.
        strategy (str): The strategy to handle outliers ('remove' or 'cap').
        multiplier (float): The IQR multiplier to determine outlier boundaries.

    Returns:
        tuple: A tuple containing the new Pandas DataFrame and an error message (if any).
    """
    df_cleaned = df.copy()
    for col in columns:
        if col not in df_cleaned.columns:
            return None, f"Error: Column '{col}' not found."

        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        if strategy == 'remove':
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        elif strategy == 'cap':
            df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)
        else:
            return None, "Error: Invalid strategy. Choose 'remove' or 'cap'."

    return df_cleaned, None

def impute_knn(df, columns, n_neighbors=5):
    """
    Imputes missing values using K-Nearest Neighbors.

    Technical: Utilizes scikit-learn's `KNNImputer`. This method models each feature
    with missing values as a function of other features, and imputes missing values
    based on the values of the k-nearest neighbors in the dataset. This operation works
    directly on a Pandas DataFrame.

    Layman: This is a smart way to fill in missing values. Instead of just using the
    average, it looks at similar data points (neighbors) to make an educated guess
    about what the missing value should be.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        columns (list): A list of numerical column names to impute.
        n_neighbors (int): The number of neighbors to use for imputation.

    Returns:
        tuple: A tuple containing the new Pandas DataFrame and an error message (if any).
    """
    try:
        if not all(col in df.columns for col in columns):
            return None, "Error: One or more specified columns do not exist."
        
        df_imputed = df.copy()
        numerical_cols = df_imputed[columns].select_dtypes(include=np.number).columns.tolist()
        
        if len(numerical_cols) != len(columns):
            return None, "Error: All columns for KNN imputation must be numerical."

        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])

        return df_imputed, None
    except Exception as e:
        return None, f"Error during KNN imputation: {e}"

def apply_regex_cleaning(df, column, pattern, replacement):
    """
    Cleans a text column using a regular expression.

    Technical: Uses pandas' `str.replace()` method with `regex=True`. This allows for
    finding and replacing substrings in a column based on a specified regex pattern.

    Layman: This is an advanced "find and replace" for your text data. You can use it
    to clean up complex text patterns, like removing special characters or extracting
    specific parts of the text.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        column (str): The name of the text column to clean.
        pattern (str): The regex pattern to search for.
        replacement (str): The string to replace the matched pattern with.

    Returns:
        tuple: A tuple containing the new Pandas DataFrame and an error message (if any).
    """
    df_cleaned = df.copy()
    if column not in df_cleaned.columns or not pd.api.types.is_string_dtype(df_cleaned[column].dtype):
        return None, f"Error: Column '{column}' is not a valid text column."
    
    try:
        df_cleaned[column] = df_cleaned[column].str.replace(pattern, replacement, regex=True)
    except Exception as e:
        return None, f"Error applying regex: {e}"

    return df_cleaned, None
    
