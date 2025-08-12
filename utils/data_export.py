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
        return None, None

def export_ipynb(df_head_html, code_log, current_stage):
    """
    Creates a Jupyter Notebook (.ipynb) file with a summary and optional code.

    Technical: This function uses the nbformat library to programmatically create
    a new Jupyter Notebook. It adds a markdown cell for a title and an HTML
    table preview of the DataFrame. It can also include Python code cells from
    the code_log. The notebook content is returned as a JSON string.

    Layman: This is a way to create a full report of your work. It creates a
    Jupyter Notebook file that you can open and run. It will include a nice
    summary of your data, and if you want, it will also include the Python
    code that was used to get to this point. This is great for sharing your
    analysis with others.

    Args:
        df_head_html (str): The HTML representation of the DataFrame's head.
        code_log (list): A list of strings, where each string is a block of Python code.
        current_stage (str): The name of the current pipeline stage.

    Returns:
        tuple: (file_content, mime_type) for the .ipynb file.
    """
    nb = new_notebook()
    
    # Add a title and description
    nb.cells.append(new_markdown_cell(f"# Data Analysis Pipeline - {current_stage}"))
    nb.cells.append(new_markdown_cell(f"### Dataset Preview\nHere is a preview of the DataFrame after the '{current_stage}' stage:\n\n{df_head_html}"))

    # Add code cells from the log
    if code_log:
        nb.cells.append(new_markdown_cell("### Code Log"))
        for code_block in code_log:
            nb.cells.append(new_code_cell(code_block))

    return nbformat.writes(nb).encode('utf-8'), 'application/x-ipynb+json'

