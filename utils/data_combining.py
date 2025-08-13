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
    instructions (like which columns to match on), and it does the work for you.

    Args:
        left_df (pd.DataFrame): The left DataFrame.
        right_df (pd.DataFrame): The right DataFrame.
        method (str): The combination method ('merge', 'concat', or 'join').
        **kwargs: Additional keyword arguments to pass to the pandas function.

    Returns:
        tuple: A tuple containing the combined DataFrame and an error message (if any).
    """
    if method == 'merge':
        try:
            combined_df = pd.merge(left_df, right_df, **kwargs)
            return combined_df, None
        except Exception as e:
            return None, f"Error during merge operation: {e}"
    elif method == 'concat':
        try:
            combined_df = pd.concat([left_df, right_df], **kwargs)
            return combined_df, None
        except Exception as e:
            return None, f"Error during concat operation: {e}"
    elif method == 'join':
        try:
            combined_df = left_df.join(right_df, **kwargs)
            return combined_df, None
        except Exception as e:
            return None, f"Error during join operation: {e}"
    else:
        return None, f"Error: Invalid combination method '{method}'. Use 'merge', 'concat', or 'join'."
