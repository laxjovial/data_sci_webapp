import pandas as pd

def combine_dataframes(left_df, right_df, method, **kwargs):
    """
    Combines two DataFrames using a specified method (merge, concat, or join).

    Technical: This function acts as a wrapper around pandas' core combination
    functions. It uses a dictionary of keyword arguments (**kwargs) to flexibly
    pass the necessary parameters (like 'on', 'how', 'left_on', 'right_on', 'axis')
    to the appropriate pandas function based on the selected 'method'.

    Layman: This is the engine that puts your two datasets together. You tell it
    'how' you want them combined (merge, concat, or join) and provide the
    instructions (like which columns to match on), and it does the work.

    Args:
        left_df (pd.DataFrame): The first (primary) DataFrame.
        right_df (pd.DataFrame): The second DataFrame.
        method (str): The combination method ('merge', 'concat', or 'join').
        **kwargs: Additional keyword arguments to pass to the pandas function.
            For 'merge' and 'join', this could include 'on', 'how', 'left_on', 'right_on'.
            For 'concat', this could include 'axis', 'ignore_index'.

    Returns:
        pd.DataFrame: The combined DataFrame.
        str: An error message if the combination fails, otherwise None.
    """
    try:
        if method == 'merge':
            if 'on' not in kwargs and ('left_on' not in kwargs or 'right_on' not in kwargs):
                return None, "Error: 'on' or 'left_on'/'right_on' key must be specified for merge."
            combined_df = pd.merge(left_df, right_df, **kwargs)
        elif method == 'concat':
            combined_df = pd.concat([left_df, right_df], **kwargs)
        elif method == 'join':
            combined_df = left_df.join(right_df, **kwargs)
        else:
            return None, "Error: Invalid combination method. Choose 'merge', 'concat', or 'join'."
        return combined_df, None
    except Exception as e:
        return None, f"An error occurred during data combination: {e}"
