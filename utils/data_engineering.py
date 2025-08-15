import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder, OrdinalEncoder, OneHotEncoder
import category_encoders as ce

def create_new_feature(df, col1, col2, operation, new_col_name):
    """
    Creates a new feature in a Dask DataFrame by combining two existing columns.

    Technical: This function leverages Dask's operator overloading to perform
    element-wise operations (+, -, *, /) on two columns, creating a new one.
    Dask automatically handles this in parallel across partitions.

    Layman: This is a simple way to create a new column by combining two other
    columns using basic math. For example, you can create a 'Total' column
    by adding 'Sales' and 'Tax' together.

    Args:
        df (dd.DataFrame): The input Dask DataFrame.
        col1 (str): The name of the first column.
        col2 (str): The name of the second column.
        operation (str): The mathematical operation to perform ('add', 'subtract',
                         'multiply', 'divide').
        new_col_name (str): The name of the new column to be created.

    Returns:
        tuple: A tuple containing the new Dask DataFrame and an error message (if any).
    """
    df_engineered = df.copy()
    if not all(col in df_engineered.columns for col in [col1, col2]):
        return None, "Error: One or both specified columns do not exist."

    try:
        if operation == 'add':
            df_engineered[new_col_name] = df_engineered[col1] + df_engineered[col2]
        elif operation == 'subtract':
            df_engineered[new_col_name] = df_engineered[col1] - df_engineered[col2]
        elif operation == 'multiply':
            df_engineered[new_col_name] = df_engineered[col1] * df_engineered[col2]
        elif operation == 'divide':
            # Check for division by zero using Dask's compute() for a small head
            if (df_engineered[col2] == 0).any().compute():
                return None, "Error: Division by zero is not allowed."
            df_engineered[new_col_name] = df_engineered[col1] / df_engineered[col2]
        else:
            return None, "Error: Invalid operation specified. Use 'add', 'subtract', 'multiply', or 'divide'."
        return df_engineered, None
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

def scale_features(df, columns, scaler_type='standard'):
    """
    Scales numerical features using a StandardScaler or MinMaxScaler on a Dask DataFrame.

    Technical: This function first computes statistics (mean, std) from the entire
    Dask DataFrame to fit the scaler. Then, it uses `map_partitions()` to apply
    the transformation to each partition in parallel, avoiding memory issues.
    This is a common pattern for using scikit-learn with Dask.

    Layman: This function standardizes the numbers in your columns so they are on a
    similar scale. This is important for many machine learning models to work
    correctly, preventing one feature with a large range from dominating others.

    Args:
        df (dd.DataFrame): The input Dask DataFrame.
        columns (list): List of numerical columns to scale.
        scaler_type (str): The type of scaler to use ('standard' or 'minmax').

    Returns:
        tuple: A tuple containing the new Dask DataFrame and an error message (if any).
    """
    df_scaled = df.copy()
    if not all(col in df_scaled.columns and pd.api.types.is_numeric_dtype(df_scaled[col].dtype) for col in columns):
        return None, "Error: All selected columns must be numeric for scaling."

    try:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            return None, "Error: Invalid scaler type. Use 'standard' or 'minmax'."

        # Fit the scaler on the computed data. This requires loading the data into memory.
        scaler.fit(df_scaled[columns].compute())
        
        # Use map_partitions to apply the transformation to each Dask partition
        df_scaled[columns] = df_scaled[columns].map_partitions(
            scaler.transform,
            meta=df_scaled[columns]._meta
        )
        return df_scaled, None
    except Exception as e:
        return None, f"An unexpected error occurred during scaling: {e}"

def apply_encoding(df, column, encoding_type, target_column=None):
    """
    Applies encoding to a categorical column in a Dask DataFrame.

    Technical: This function first computes the unique categories from the entire
    Dask DataFrame to fit the encoder. It then uses `map_partitions` to apply the
    transformation to each partition, which is more memory-efficient than a full `compute()`.
    Note: TargetEncoder from `category_encoders` may require computing the full data.

    Layman: This turns text categories, like 'Red', 'Green', 'Blue', into
    numbers that a machine learning model can understand. This is a crucial step
    for using categorical data in most models.

    Args:
        df (dd.DataFrame): The input Dask DataFrame.
        column (str): The column to encode.
        encoding_type (str): The type of encoding to apply ('one-hot', 'label', 'ordinal', 'target').
        target_column (str, optional): The name of the target column, required for target encoding.

    Returns:
        tuple: A tuple containing the new Dask DataFrame and an error message (if any).
    """
    df_encoded = df.copy()
    if column not in df_encoded.columns:
        return None, f"Error: Column '{column}' not found."

    try:
        if encoding_type == 'one-hot':
            # This is complex with Dask; it's often easier to compute, encode, and then re-Dask
            pandas_df = df_encoded.compute()
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_cols = encoder.fit_transform(pandas_df[[column]])
            encoded_df = pd.DataFrame(encoded_cols, index=pandas_df.index, columns=encoder.get_feature_names_out([column]))
            # Concatenate and convert back to Dask
            df_encoded = pd.concat([pandas_df.drop(columns=column), encoded_df], axis=1)
            return dd.from_pandas(df_encoded, npartitions=df.npartitions), None

        elif encoding_type == 'label':
            # Dask's .astype('category') handles this efficiently.
            df_encoded[f'{column}_encoded'] = df_encoded[column].astype('category').cat.codes
            return df_encoded, None
        
        elif encoding_type == 'ordinal':
            # This also leverages Dask's built-in categorical type
            df_encoded[f'{column}_ordinal'] = df_encoded[column].astype('category').cat.as_ordered().cat.codes
            return df_encoded, None

        elif encoding_type == 'target':
            if target_column is None or target_column not in df_encoded.columns:
                return None, "Error: Target encoding requires a valid target column."
            
            # TargetEncoder needs to see the full data and is not Dask-native.
            # This requires a full compute.
            pandas_df = df_encoded.compute()
            encoder = ce.TargetEncoder(cols=[column])
            pandas_df[f'{column}_target_encoded'] = encoder.fit_transform(pandas_df[column], pandas_df[target_column])
            return dd.from_pandas(pandas_df, npartitions=df.npartitions), None

        else:
            return None, "Error: Invalid encoding type. Use 'one-hot', 'label', 'ordinal', or 'target'."
    except Exception as e:
        return None, f"An unexpected error occurred during encoding: {e}"

def bin_column(df, column, bins):
    """
    Bins a numerical column into discrete intervals on a Dask DataFrame.

    Technical: This function uses `pandas.cut` under the hood. For it to work with Dask,
    we must compute the column to get the bin edges, or provide them manually. For
    simplicity, we'll use `map_partitions` with pre-calculated bins.

    Layman: This is useful for turning a column with a wide range of numbers
    (like 'Age') into a column with a few groups (like 'Child', 'Teen', 'Adult').
    This can make data easier to work with or visualize.

    Args:
        df (dd.DataFrame): The input Dask DataFrame.
        column (str): The name of the column to bin.
        bins (int or sequence): The number of bins to create, or the bin edges.

    Returns:
        tuple: A tuple containing the new Dask DataFrame and an error message (if any).
    """
    df_binned = df.copy()
    if column not in df_binned.columns or not pd.api.types.is_numeric_dtype(df_binned[column].dtype):
        return None, f"Error: Column '{column}' must be numeric for binning."
    
    try:
        # A simple approach for Dask is to compute the bins once and apply to all partitions.
        bins_values = np.histogram(df_binned[column].compute(), bins=bins)[1]

        def bin_partition(partition):
            return pd.cut(partition[column], bins=bins_values, include_lowest=True, right=False)

        # Use map_partitions to apply the binning to each partition
        df_binned[f'{column}_binned'] = df_binned.map_partitions(
            bin_partition, 
            meta=(f'{column}_binned', 'category')
        )
        return df_binned, None
    except Exception as e:
        return None, f"An unexpected error occurred during binning: {e}"

def create_polynomial_features(df, column, degree=2, interaction_only=False):
    """
    Creates polynomial and interaction features from a numerical column in a Dask DataFrame.

    Technical: This function first computes the column to fit the `PolynomialFeatures`
    object. Then, it uses `map_partitions` to apply the transformation across each
    partition. This is a memory-efficient way to use this scikit-learn transformer with Dask.

    Layman: This is a way to create new, more complex features from an existing one. For
    example, from an 'Age' column, you could create 'Age^2' and 'Age^3' features. These
    can help a model capture non-linear relationships in the data.

    Args:
        df (dd.DataFrame): The input Dask DataFrame.
        column (str): The numerical column to transform.
        degree (int): The degree of the polynomial features.
        interaction_only (bool): If True, only interaction features are produced.

    Returns:
        tuple: A tuple containing the new Dask DataFrame and an error message (if any).
    """
    df_poly = df.copy()
    if column not in df_poly.columns or not pd.api.types.is_numeric_dtype(df_poly[column].dtype):
        return None, "Error: Column for polynomial features must be numeric."

    try:
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=interaction_only)
        
        # Fit the transformer on a computed version of the column
        poly.fit(df_poly[[column]].compute())
        
        # Create a new Dask DataFrame by mapping the transformation across partitions
        # This requires careful metadata handling
        def apply_poly(partition):
            transformed_data = poly.transform(partition[[column]])
            return pd.DataFrame(transformed_data, index=partition.index, columns=poly.get_feature_names_out([column]))

        poly_dd = df_poly.map_partitions(
            apply_poly,
            meta=pd.DataFrame(columns=poly.get_feature_names_out([column]), dtype=float)
        )

        # Concatenate the new features with the original DataFrame
        df_poly = dd.concat([df_poly, poly_dd], axis=1)

        return df_poly, None
    except Exception as e:
        return None, f"An unexpected error occurred during polynomial feature creation: {e}"