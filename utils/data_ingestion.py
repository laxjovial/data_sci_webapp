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
    Loads a dataset from a file path or URL.

    Technical: The function handles different sources. For 'upload', it reads
    the file from a local path. For 'url', it makes an HTTP GET request,
    then uses pandas' read_csv() to parse the content directly from a
    text stream.

    Layman: This is the "open file" button for your application. It can
    read data from a file you've uploaded or a link you've provided.

    Args:
        source_type (str): The source of the data ('upload', 'url').
        source_path_or_file (str): The local file path or the URL to the data.

    Returns:
        tuple: A tuple containing the loaded DataFrame and an error message (if any).
    """
    df = None
    error = None

    if source_type == 'upload':
        try:
            # Assuming CSV for now, but could be extended to other formats
            df = pd.read_csv(source_path_or_file)
        except FileNotFoundError:
            error = f"Error: File not found at {source_path_or_file}."
        except Exception as e:
            error = f"Error reading file: {e}"
            
    elif source_type == 'url':
        url = _convert_gdrive_url(source_path_or_file)
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Check if the response content is a zip file
            if 'zip' in response.headers.get('Content-Type', ''):
                return None, "Error: The provided URL points to a zip file. Please provide a direct link to a data file like CSV or Excel."
                
            file_content = io.StringIO(response.text)
            # Try to infer the file type, assuming CSV if not specified
            df = pd.read_csv(file_content)

        except requests.exceptions.RequestException as e:
            error = f"Error accessing URL: {e}"
        except pd.errors.ParserError as e:
            error = f"Error parsing data from URL: {e}. Please ensure the URL points to a valid CSV file."
        except Exception as e:
            error = f"An unexpected error occurred: {e}"
            
    else:
        error = f"Error: Invalid source type '{source_type}'."
        
    return df, error
