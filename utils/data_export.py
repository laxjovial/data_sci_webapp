import pandas as pd
import io

def export_dataframe(df, file_format):
    """
    Exports a pandas DataFrame to various file formats.

    Technical: This function takes a DataFrame and a format string, then uses
    pandas' built-in methods to convert the DataFrame into a binary or string
    representation for a file download. It now supports CSV, Excel, JSON, and Parquet.

    Layman: This is the "save" button for your data. You can choose to save your
    data as a CSV, Excel, JSON, or Parquet file, and it will be prepared for download.

    Args:
        df (pd.DataFrame): The input DataFrame.
        file_format (str): The desired file format ('csv', 'excel', 'json', 'parquet').

    Returns:
        tuple: A tuple containing the file data (as bytes) and an error message (if any).
    """
    try:
        if file_format == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False)
            return output.getvalue().encode('utf-8'), None
        elif file_format == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            return output.getvalue(), None
        elif file_format == 'json':
            output = io.StringIO()
            df.to_json(output, orient='records', indent=4)
            return output.getvalue().encode('utf-8'), None
        elif file_format == 'parquet':
            output = io.BytesIO()
            df.to_parquet(output, index=False)
            return output.getvalue(), None
        else:
            return None, f"Error: Invalid file format '{file_format}'."
    except Exception as e:
        return None, f"An unexpected error occurred during export: {e}"
