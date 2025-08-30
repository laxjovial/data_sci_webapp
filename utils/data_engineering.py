import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder, OneHotEncoder
import category_encoders as ce

def create_new_feature(df, col1, col2, operation, new_col_name):
    """
    Creates a new feature in a Pandas DataFrame by combining two existing columns.

    Technical: This function performs element-wise arithmetic operations (+, -, *, /)
    on two columns to create a new one. It includes a check for division by zero.

    Layman: This is a simple way to create a new column by combining two other
    columns using basic math. For example, you can create a 'Total' column
    by adding 'Sales' and 'Tax' together.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        col1 (str): The name of the first column.
        col2 (str): The name of the second column.
        operation (str): The mathematical operation to perform ('add', 'subtract',
                         'multiply', 'divide').
        new_col_name (str): The name of the new column to be created.

    Returns:
        tuple: A tuple containing the new Pandas DataFrame and an error message (if any).
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
            if (df_engineered[col2] == 0).any():
                return None, "Error: Division by zero is not allowed."
            df_engineered[new_col_name] = df_engineered[col1] / df_engineered[col2]
        else:
            return None, "Error: Invalid operation specified. Use 'add', 'subtract', 'multiply', or 'divide'."
        return df_engineered, None
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

def scale_features(df, columns, scaler_type='standard'):
    """
    Scales numerical features using a StandardScaler or MinMaxScaler on a Pandas DataFrame.

    Technical: This function applies a scikit-learn scaler (`StandardScaler` or `MinMaxScaler`)
    to the specified numerical columns of a Pandas DataFrame. The scaling is done in-place.

    Layman: This function standardizes the numbers in your columns so they are on a
    similar scale. This is important for many machine learning models to work
    correctly, preventing one feature with a large range from dominating others.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        columns (list): List of numerical columns to scale.
        scaler_type (str): The type of scaler to use ('standard' or 'minmax').

    Returns:
        tuple: A tuple containing the new Pandas DataFrame and an error message (if any).
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

        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        return df_scaled, None
    except Exception as e:
        return None, f"An unexpected error occurred during scaling: {e}"

def apply_encoding(df, column, encoding_type, target_column=None):
    """
    Applies encoding to a categorical column in a Pandas DataFrame.

    Technical: This function provides several common encoding methods for categorical
    data, including one-hot, label, and target encoding, using pandas and
    scikit-learn libraries.

    Layman: This turns text categories, like 'Red', 'Green', 'Blue', into
    numbers that a machine learning model can understand. This is a crucial step
    for using categorical data in most models.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        column (str): The column to encode.
        encoding_type (str): The type of encoding to apply ('one-hot', 'label', 'target').
        target_column (str, optional): The name of the target column, required for target encoding.

    Returns:
        tuple: A tuple containing the new Pandas DataFrame and an error message (if any).
    """
    df_encoded = df.copy()
    if column not in df_encoded.columns:
        return None, f"Error: Column '{column}' not found."

    try:
        if encoding_type == 'one-hot':
            return pd.get_dummies(df_encoded, columns=[column]), None

        elif encoding_type == 'label':
            le = LabelEncoder()
            df_encoded[f'{column}_encoded'] = le.fit_transform(df_encoded[column])
            return df_encoded, None

        elif encoding_type == 'target':
            if target_column is None or target_column not in df_encoded.columns:
                return None, "Error: Target encoding requires a valid target column."
            
            encoder = ce.TargetEncoder(cols=[column])
            df_encoded[f'{column}_target_encoded'] = encoder.fit_transform(df_encoded[column], df_encoded[target_column])
            return df_encoded, None

        else:
            return None, "Error: Invalid encoding type. Use 'one-hot', 'label', or 'target'."
    except Exception as e:
        return None, f"An unexpected error occurred during encoding: {e}"

def bin_column(df, column, bins):
    """
    Bins a numerical column into discrete intervals on a Pandas DataFrame.

    Technical: This function uses `pandas.cut` to segment and sort data values into bins.
    This is useful for going from a continuous variable to a categorical variable.

    Layman: This is useful for turning a column with a wide range of numbers
    (like 'Age') into a column with a few groups (like 'Child', 'Teen', 'Adult').
    This can make data easier to work with or visualize.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        column (str): The name of the column to bin.
        bins (int): The number of equal-width bins to create.

    Returns:
        tuple: A tuple containing the new Pandas DataFrame and an error message (if any).
    """
    df_binned = df.copy()
    if column not in df_binned.columns or not pd.api.types.is_numeric_dtype(df_binned[column].dtype):
        return None, f"Error: Column '{column}' must be numeric for binning."
    
    try:
        df_binned[f'{column}_binned'] = pd.cut(df_binned[column], bins=int(bins))
        return df_binned, None
    except Exception as e:
        return None, f"An unexpected error occurred during binning: {e}"

def create_polynomial_features(df, column, degree=2, interaction_only=False):
    """
    Creates polynomial and interaction features from a numerical column in a Pandas DataFrame.

    Technical: This function uses scikit-learn's `PolynomialFeatures` to generate
    new features. It can create both polynomial features (e.g., x^2, x^3) and
    interaction features between columns if multiple columns are provided.

    Layman: This is a way to create new, more complex features from an existing one. For
    example, from an 'Age' column, you could create 'Age^2' and 'Age^3' features. These
    can help a model capture non-linear relationships in the data.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        column (str): The numerical column to transform.
        degree (int): The degree of the polynomial features.
        interaction_only (bool): If True, only interaction features are produced.

    Returns:
        tuple: A tuple containing the new Pandas DataFrame and an error message (if any).
    """
    df_poly = df.copy()
    if column not in df_poly.columns or not pd.api.types.is_numeric_dtype(df_poly[column].dtype):
        return None, "Error: Column for polynomial features must be numeric."

    try:
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=interaction_only)
        
        # Reshape data for the transformer
        poly_features = poly.fit_transform(df_poly[[column]])

        # Create a new DataFrame with the polynomial features
        poly_df = pd.DataFrame(poly_features, index=df_poly.index, columns=poly.get_feature_names_out([column]))
        
        # Concatenate the new features with the original DataFrame
        df_poly = pd.concat([df_poly, poly_df], axis=1)

        return df_poly, None
    except Exception as e:
        return None, f"An unexpected error occurred during polynomial feature creation: {e}"