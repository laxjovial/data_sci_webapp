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
        source_path_or_file (str or werkzeug.FileStorage): The URL string or the uploaded file object.

    Returns:
        tuple: A pandas DataFrame and an error message (if any).
    """
    df = None
    error_message = None
    try:
        if source_type == 'url':
            # Handle GitHub raw links
            if 'github.com' in source_path_or_file:
                source_path_or_file = source_path_or_file.replace(
                    'github.com', 'raw.githubusercontent.com').replace(
                    '/blob/', '/')
            # Handle Google Drive URLs
            elif 'drive.google.com' in source_path_or_file:
                source_path_or_file = _convert_gdrive_url(source_path_or_file)

            response = requests.get(source_path_or_file)
            response.raise_for_status() # Raise an exception for bad status codes
            
            content = io.StringIO(response.text)
            
            # Infer file type from URL or content
            file_extension = os.path.splitext(urllib.parse.urlparse(source_path_or_file).path)[1]
            if file_extension == '.csv' or response.headers['content-type'] == 'text/csv':
                df = pd.read_csv(content)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(io.BytesIO(response.content))
            elif file_extension == '.json':
                df = pd.read_json(content)
            else:
                error_message = f"Unsupported file type from URL: {file_extension}"
                return None, error_message

        elif source_type == 'upload':
            filename = source_path_or_file.filename
            if filename.endswith('.csv'):
                df = pd.read_csv(source_path_or_file)
            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(source_path_or_file)
            elif filename.endswith('.json'):
                df = pd.read_json(source_path_or_file)
            else:
                error_message = f"Unsupported file type uploaded: {filename}"
                return None, error_message

        else:
            error_message = "Invalid source type specified."
            return None, error_message

        return df, None
    except requests.exceptions.RequestException as e:
        return None, f"Network error during data ingestion: {e}"
    except Exception as e:
        return None, f"An error occurred during data ingestion: {e}"
