# utils/data_ingestion.py

import pandas as pd
import requests
from io import StringIO
from urllib.parse import urlparse

def load_data(source_type, source_path):
    """
    Loads a dataset from various sources (local file, URL, etc.).
    
    This function acts as a centralized data ingestion utility, handling different
    data formats and sources with robust error handling.
    
    Technical: Uses pandas.read_csv for CSVs and pandas.read_excel for Excel files. 
    For URLs, it first fetches the content using the requests library and then 
    loads it into a DataFrame from an in-memory text buffer. It uses try-except 
    blocks to gracefully handle common errors like FileNotFoundError and invalid URLs.
    
    Layman: This is the part of our app that grabs your data, whether it's from 
    a file on your computer, a link from the internet, or a cloud service. It's
    built to be smart and handle different kinds of files and potential problems.
    
    Args:
        source_type (str): The type of data source (e.g., 'local', 'url').
        source_path (str): The path to the data source (file path or URL).
    
    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
        str: An error message if data loading fails, otherwise None.
    """
    df = None
    error_message = None

    try:
        if source_type == 'local':
            if source_path.endswith('.csv'):
                df = pd.read_csv(source_path)
            elif source_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(source_path)
            else:
                error_message = "Unsupported file format. Please use .csv or .xlsx."
        
        elif source_type == 'url':
            # Check if the URL is from GitHub raw content
            parsed_url = urlparse(source_path)
            if 'github.com' in parsed_url.netloc and 'raw' not in parsed_url.path:
                source_path = source_path.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            
            response = requests.get(source_path)
            response.raise_for_status() # Raise an exception for bad status codes
            
            # Use StringIO to read the content as if it were a local file
            if source_path.endswith('.csv'):
                df = pd.read_csv(StringIO(response.text))
            elif source_path.endswith(('.xls', '.xlsx')):
                # Excel files need to be handled as binary content
                df = pd.read_excel(response.content)
            else:
                error_message = "Unsupported URL file format. Please link to a .csv or .xlsx file."
        
        elif source_type == 'gdrive':
            # A more robust G-drive integration would use an API,
            # but for this app, we'll assume a publicly shared 'csv' link.
            # Example: https://docs.google.com/spreadsheets/d/1Xl.../edit?usp=sharing
            # needs to be converted to: https://docs.google.com/spreadsheets/d/1Xl.../export?format=csv
            if 'docs.google.com/spreadsheets' in source_path:
                file_id = source_path.split('/d/')[1].split('/')[0]
                export_url = f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv'
                response = requests.get(export_url)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text))
            else:
                error_message = "Invalid Google Drive link. Please ensure it's a shared spreadsheet link."
        
        else:
            error_message = "Unsupported data source type."

    except FileNotFoundError:
        error_message = f"Error: The file at path '{source_path}' was not found."
    except pd.errors.ParserError:
        error_message = "Error: Could not parse the file. Please check the file format."
    except requests.exceptions.RequestException as e:
        error_message = f"Error connecting to the URL: {e}"
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
    
    return df, error_message



def get_dataframe_summary(df):
    """
    Generates a descriptive summary of the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to summarize.

    Returns:
        tuple: (df_head_html, df_info_str, df_desc_html, columns, unique_values)
    """
    info_buffer = pd.io.common.StringIO()
    df.info(buf=info_buffer)
    info_str = info_buffer.getvalue()

    desc_html = df.describe(include='all').to_html(classes='table table-striped table-bordered')
    unique_values = {col: df[col].unique().tolist()[:10] for col in df.columns}
    df_head_html = df.head().to_html(classes='table table-striped table-bordered')

    return df_head_html, info_str, desc_html, df.columns.tolist(), unique_values
