# utils/data_export.py
import pandas as pd
import io

def export_dataframe(df, file_format):
    """
    Exports a pandas DataFrame to various file formats.

    Technical: This function takes a DataFrame and a format string, then uses
    pandas' built-in methods to convert the DataFrame into a binary or string
    representation for a file download. It supports CSV and Excel formats.

    Layman: This is the "save" button for your data. You tell it if you
    want a CSV file or an Excel file, and it prepares your data for
    downloading in that format.

    Args:
        df (pd.DataFrame): The input DataFrame.
        file_format (str): The desired file format ('csv', 'excel').

    Returns:
        tuple: A tuple containing the file data (as bytes) and an error message (if any).
    """
    try:
        if file_format == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return output.getvalue().encode('utf-8'), None
        elif file_format == 'excel':
            output = io.BytesIO()
            df.to_excel(output, index=False)
            output.seek(0)
            return output.getvalue(), None
        else:
            return None, f"Error: Invalid file format '{file_format}'. Use 'csv' or 'excel'."
    except Exception as e:
        return None, f"An unexpected error occurred during export: {e}"
