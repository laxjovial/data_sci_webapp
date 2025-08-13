# utils/data_export.py
import pandas as pd
import io
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import base64

def export_dataframe(df, file_format):
    """
    Exports a pandas DataFrame to various file formats.

    Technical: This function takes a DataFrame and a format string, then uses
    pandas' built-in methods to convert the DataFrame into a binary or string
    representation for a file download. It supports CSV and Excel formats.

    Layman: This is the "save" button for your data. You tell it if you want to
    save your cleaned data as a simple text file (.csv) or an Excel spreadsheet
    (.xlsx), and it prepares the file for you to download.

    Args:
        df (pd.DataFrame): The DataFrame to export.
        file_format (str): The desired export format ('csv', 'xlsx').

    Returns:
        tuple: (file_content, mime_type) or (None, None) on error.
    """
    try:
        if file_format == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False)
            return output.getvalue().encode('utf-8'), 'text/csv'
        elif file_format == 'xlsx':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            output.seek(0)
            return output.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            return None, "Error: Unsupported file format specified. Use 'csv' or 'xlsx'."
    except Exception as e:
        return None, f"An error occurred during export: {e}"
