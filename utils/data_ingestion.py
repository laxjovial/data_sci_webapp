# utils/data_ingestion.py
import pandas as pd
import requests
import io
import re
import urllib.parse
import os

def _convert_gdrive_url(url):
    """
    Converts a standard Google Drive sharing URL into a direct download URL.
    This works for both Google Sheets and uploaded files.
    
    Technical: The function detects different Google Drive URL formats (e.g., 
    docs.google.com/spreadsheets or drive.google.com/file) and transforms them
    into a format that forces a direct download. This bypasses the preview page
    that a regular shared link would open.

    Args:
        url (str): The original Google Drive URL.

    Returns:
        str: The converted direct download URL.
    """
    # Pattern for Google Sheets
    gdrive_sheets_pattern = r'https:\/\/docs\.google\.com\/spreadsheets\/d\/([a-zA-Z0-9_-]+)\/edit#gid=(\d+)'
    
    # Pattern for uploaded files in Google Drive
    gdrive_file_pattern = r'https:\/\/drive\.google\.com\/file\/d\/([a-zA-Z0-9_-]+)\/view'

    sheets_match = re.search(gdrive_sheets_pattern, url)
    file_match = re.search(gdrive_file_pattern, url)
    
    if sheets_match:
        doc_id = sheets_match.group(1)
        gid = sheets_match.group(2)
        return f'https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={gid}'
    
    elif file_match:
        file_id = file_match.group(1)
        return f'https://drive.google.com/uc?id={file_id}&export=download'
    
    return url

def load_data(source_type, source_path_or_file):
    """
    Loads a dataset from various sources (URL, local file upload) into a DataFrame.
    
    Technical: This function acts as the primary data ingestion point. It uses 
    conditional logic to determine the data source type. For URLs, it first 
    converts them to a direct-download format if they're from GitHub or Google 
    Drive before making an HTTP request. For file uploads, it reads the file-like
    object provided by Flask and uses pandas to parse it based on file extension.

    Layman: This is the app's "waiter." You tell it where your data is—a web link
    or a file on your computer—and it goes and fetches it for you, making sure
    it's in the right format for the rest of the app to use.

    Args:
        source_type (str): The type of data source ('url', 'upload').
        source_path_or_file (str or werkzeug.FileStorage): The URL string or
                                                             the uploaded file object.

    Returns:
        tuple: A pandas DataFrame and an error message (if any).
    """
    df = None
    error_message = None
    
    try:
        if source_type == 'url':
            # Handle GitHub raw links
            if 'github.com' in source_path_or_file and 'raw.githubusercontent.com' not in source_path_or_file:
                source_path_or_file = source_path_or_file.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            
            # Handle Google Drive links
            if 'drive.google.com' in source_path_or_file or 'docs.google.com' in source_path_or_file:
                source_path_or_file = _convert_gdrive_url(source_path_or_file)
            
            # Make the request and read into a DataFrame
            response = requests.get(source_path_or_file)
            response.raise_for_status() # Raise an error for bad status codes
            
            content_type = response.headers.get('Content-Type', '')
            
            if 'csv' in content_type or source_path_or_file.endswith('.csv'):
                df = pd.read_csv(io.StringIO(response.text))
            elif 'excel' in content_type or source_path_or_file.endswith('.xlsx') or source_path_or_file.endswith('.xls'):
                df = pd.read_excel(io.BytesIO(response.content))
            else:
                error_message = f"Unsupported file type from URL: {source_path_or_file}"
        
        elif source_type == 'upload':
            filename = source_path_or_file.filename
            if filename.endswith('.csv'):
                df = pd.read_csv(source_path_or_file)
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(source_path_or_file)
            else:
                error_message = f"Unsupported file type for upload: {filename}. Please upload a CSV or Excel file."
                
    except requests.exceptions.RequestException as e:
        error_message = f"Failed to retrieve data from URL: {e}"
    except Exception as e:
        error_message = f"An unexpected error occurred during data loading: {e}"

    return df, error_message
    
def get_dataframe_summary(df):
    """
    Generates a summary of the DataFrame for display.

    Technical: This is a utility function that generates HTML and string 
    representations of the DataFrame's head, info, and descriptive statistics. 
    It also extracts column names and unique values for categorical columns, 
    making it easier to display these details in the web interface.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: (df_head_html, df_info_str, df_desc_html, columns, unique_values_dict)
    """
    df_head_html = df.head().to_html(classes=['table', 'table-striped', 'table-sm'])
    
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    
    df_desc_html = df.describe().to_html(classes=['table', 'table-striped', 'table-sm'])
    columns = df.columns.tolist()
    
    unique_values = {}
    for col in df.columns:
        if df[col].nunique() < 20 and df[col].dtype == 'object':
            unique_values[col] = df[col].unique().tolist()
    
    return df_head_html, info_str, df_desc_html, columns, unique_values
