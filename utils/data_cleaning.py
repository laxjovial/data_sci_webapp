import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def handle_missing_values(df, columns, strategy):
    """Handles missing values in specified columns of a Dask DataFrame."""
    df_cleaned = df.copy()
    if not all(col in df_cleaned.columns for col in columns):
        return None, "Error: One or more specified columns do not exist."
    
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna(subset=columns)
    else:
        for col in columns:
            if strategy == 'mean':
                fill_value = df_cleaned[col].mean().compute()
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
            elif strategy == 'median':
                # Dask doesn't have a direct median, so we approximate
                fill_value = df_cleaned[col].quantile(0.5).compute()
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
            elif strategy == 'mode':
                fill_value = df_cleaned[col].mode().compute()[0]
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    return df_cleaned, None

def rename_column(df, old_col, new_col):
    """Renames a column in a Dask DataFrame."""
    return df.rename(columns={old_col: new_col}), None

def convert_dtype(df, column, new_type):
    """Converts a Dask DataFrame column to a new data type."""
    try:
        df[column] = df[column].astype(new_type)
        return df, None
    except Exception as e:
        return None, f"Error converting dtype: {e}"

def remove_duplicates(df):
    """Removes duplicate rows from a Dask DataFrame."""
    return df.drop_duplicates(), None

def standardize_text(df, columns):
    """Standardizes text columns in a Dask DataFrame."""
    df_cleaned = df.copy()
    for col in columns:
        if col not in df_cleaned.columns:
            return None, f"Error: Column '{col}' not found."
        df_cleaned[col] = df_cleaned[col].str.lower().str.strip()
    return df_cleaned, None

def handle_outliers_iqr(df, columns, strategy='remove', multiplier=1.5):
    """
    Handles outliers in numerical columns using the Interquartile Range (IQR) method on a Dask DataFrame.

    Technical: Calculates the IQR (Q3 - Q1) for each specified column. Outliers are
    identified as data points that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    The function can either remove these rows or cap the values at the outlier boundaries.

    Layman: This function finds and deals with extreme values (outliers) in your data.
    You can choose to either completely remove the rows containing these outliers or
    replace the outlier values with the nearest "normal" value.

    Args:
        df (dd.DataFrame): The input Dask DataFrame.
        columns (list): A list of numerical column names to process.
        strategy (str): The strategy to handle outliers ('remove' or 'cap').
        multiplier (float): The IQR multiplier to determine outlier boundaries.

    Returns:
        tuple: A tuple containing the new Dask DataFrame and an error message (if any).
    """
    df_cleaned = df.copy()
    for col in columns:
        if col not in df_cleaned.columns:
            return None, f"Error: Column '{col}' not found."

        # The quantiles must be computed to be used in the filter
        Q1 = df_cleaned[col].quantile(0.25).compute()
        Q3 = df_cleaned[col].quantile(0.75).compute()
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
    based on the values of the k-nearest neighbors in the dataset. NOTE: This operation
    requires converting the Dask DataFrame to a Pandas DataFrame by calling `.compute()`,
    which can be very slow and memory-intensive on large datasets.

    Layman: This is a smart way to fill in missing values. Instead of just using the
    average, it looks at similar data points (neighbors) to make an educated guess
    about what the missing value should be.

    Args:
        df (dd.DataFrame): The input Dask DataFrame.
        columns (list): A list of numerical column names to impute.
        n_neighbors (int): The number of neighbors to use for imputation.

    Returns:
        tuple: A tuple containing the new Dask DataFrame and an error message (if any).
    """
    try:
        # Check if all columns exist and are numerical before computing
        if not all(col in df.columns for col in columns):
            return None, "Error: One or more specified columns do not exist."
        
        # Select and compute the necessary columns for scikit-learn
        pandas_df = df[columns].compute()
        
        # Check for non-numerical columns after computing
        numerical_cols = pandas_df.select_dtypes(include=np.number).columns.tolist()
        if len(numerical_cols) != len(columns):
            return None, "Error: All columns for KNN imputation must be numerical."

        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(pandas_df)

        # Create a dask dataframe from the imputed pandas dataframe
        imputed_dd = dd.from_pandas(pd.DataFrame(imputed_data, columns=columns), npartitions=df.npartitions)
        
        # Update the original dask dataframe with the imputed columns
        # To handle column order, we drop and then concat
        df = df.drop(columns=columns)
        df = dd.concat([df, imputed_dd], axis=1)

        return df, None
    except Exception as e:
        return None, f"Error during KNN imputation: {e}"

def apply_regex_cleaning(df, column, pattern, replacement):
    """
    Cleans a text column using a regular expression.

    Technical: Uses pandas' `str.replace()` method with `regex=True`, which is supported
    by Dask. This allows for finding and replacing substrings in a column based on a
    specified regex pattern.

    Layman: This is an advanced "find and replace" for your text data. You can use it
    to clean up complex text patterns, like removing special characters or extracting
    specific parts of the text.

    Args:
        df (dd.DataFrame): The input Dask DataFrame.
        column (str): The name of the text column to clean.
        pattern (str): The regex pattern to search for.
        replacement (str): The string to replace the matched pattern with.

    Returns:
        tuple: A tuple containing the new Dask DataFrame and an error message (if any).
    """
    df_cleaned = df.copy()
    if column not in df_cleaned.columns or not pd.api.types.is_string_dtype(df_cleaned[column].dtype):
        return None, f"Error: Column '{column}' is not a valid text column."
    
    try:
        df_cleaned[column] = df_cleaned[column].str.replace(pattern, replacement, regex=True)
    except Exception as e:
        return None, f"Error applying regex: {e}"

    return df_cleaned, None
    
