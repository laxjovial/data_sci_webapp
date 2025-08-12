# utils/data_engineering.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

def apply_encoding(df, columns, encoding_type):
    """
    Applies various encoding techniques to specified categorical columns.

    Technical: This function provides a unified interface for several encoding
    methods. Label Encoding is for binary or ordinal data. Ordinal Encoding
    is for converting categorical strings to integers. One-Hot Encoding
    creates new binary columns for each category. Frequency Encoding replaces
    categories with their frequency in the dataset.

    Layman: This is where we turn text-based categories (like 'car' or 'radio/TV')
    into numbers that a computer can understand. We have different ways to do this,
    depending on whether the categories have a natural order or not.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to encode.
        encoding_type (str): The type of encoding to apply.
                             ('one_hot', 'label', 'ordinal', 'frequency')

    Returns:
        tuple: The DataFrame with encoded columns and an error message (if any).
    """
    df_encoded = df.copy()
    error_message = None

    if encoding_type == 'label':
        # Label Encoding is best for binary categories like 'male'/'female'
        # or when there's an implicit order
        for col in columns:
            if df_encoded[col].nunique() <= 2:
                try:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                except Exception as e:
                    error_message = f"Error applying Label Encoding to '{col}': {e}"
                    return df, error_message
            else:
                error_message = "Label Encoding is recommended for binary columns. Use Ordinal or One-Hot for others."
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
        # It creates a new column for each unique value.
        try:
            df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
        except Exception as e:
            error_message = f"Error applying One-Hot Encoding: {e}"
            return df, error_message
            
    elif encoding_type == 'frequency':
        # Frequency Encoding replaces categories with the proportion of their occurrence.
        # It can be useful for tree-based models.
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


def remove_outliers(data, columns):
    """
    Removes outliers from the specified columns of the dataframe using the IQR method.
    
    Technical: This function computes the Interquartile Range (IQR) for each
    numeric column and removes rows where the values fall outside the
    standard 1.5 * IQR from the first and third quartiles (Q1, Q3).

    Layman: This is a way to get rid of extreme or unusual values that might
    skew your analysis. We find a typical range for the data and remove any
    data points that are too far away from that range.

    Parameters:
    data (pd.DataFrame): The dataframe to remove outliers from.
    columns (list): List of columns to check for outliers.

    Returns:
    pd.DataFrame: Dataframe with outliers removed.
    """
    df_cleaned = data.copy()
    for col in columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

    return df_cleaned, None
