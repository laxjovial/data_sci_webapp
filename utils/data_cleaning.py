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
    for col in columns:
        df[col] = df[col].str.lower().str.strip()
    return df, None

def handle_outliers_iqr(df, columns, strategy='remove', multiplier=1.5):
    """Handles outliers in a Dask DataFrame using the IQR method."""
    df_cleaned = df.copy()
    for col in columns:
        Q1 = df_cleaned[col].quantile(0.25).compute()
        Q3 = df_cleaned[col].quantile(0.75).compute()
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        if strategy == 'remove':
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        elif strategy == 'cap':
            df_cleaned[col] = df_cleaned[col].clip(lower_bound, upper_bound)
    return df_cleaned, None

def impute_knn(df, columns, n_neighbors=5):
    """
    Imputes missing values using KNN. This requires computing the data.
    NOTE: This operation can be very slow and memory-intensive on large datasets.
    """
    try:
        # Compute the necessary columns to pass to scikit-learn
        pandas_df = df[columns].compute()
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(pandas_df)

        # Create a dask dataframe from the imputed pandas dataframe
        imputed_dd = dd.from_pandas(pd.DataFrame(imputed_data, columns=columns), npartitions=df.npartitions)

        # Update the original dask dataframe
        df = df.drop(columns=columns)
        df = dd.concat([df, imputed_dd], axis=1)
        return df, None
    except Exception as e:
        return None, f"Error during KNN imputation: {e}"
