# utils/data_ingestion.py
import pandas as pd
import requests
import io
import re
import os

def _get_file_extension(path):
    """Extracts the file extension from a path or URL."""
    try:
        # For URLs, strip query parameters first
        if '?' in path:
            path = path.split('?')[0]
        return os.path.splitext(path)[1].lower()
    except:
        return ""

def _convert_gdrive_url(url):
    """Converts a Google Drive sharing URL into a direct download URL."""
    gdrive_sheets_pattern = r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)/edit#gid=(\d+)'
    gdrive_file_pattern = r'https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)/view'
    
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

def _convert_github_url(url):
    """Converts a GitHub file URL to a raw content URL."""
    if 'github.com' in url and '/blob/' in url:
        return url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
    return url

def load_data(source_path_or_file, source_type='upload', file_type=None, delimiter=',', encoding='utf-8'):
    """
    Loads a dataset from a file path or URL as a Pandas DataFrame.
    
    Technical: This function handles both local file uploads and URL-based data.
    For URLs, it converts them to raw download links (for Google Drive and GitHub)
    and reads the content directly into memory without saving a temporary file. It then
    uses pandas' `read_*` functions to parse the data. This is suitable for datasets
    that can comfortably fit into the application's memory.

    Layman: This function opens your data file, whether it's from your computer or a web link
    (including Google Drive and GitHub). It automatically figures out if it's a CSV,
    Excel, JSON, or Parquet file and reads it for you.

    Args:
        source_path_or_file (str or file-like object): The local file path, URL, or uploaded file object.
        source_type (str): The source of the data ('upload', 'url').
        file_type (str, optional): The type of file ('csv', 'excel', 'json', 'parquet').
                                   If None, it will be inferred from the filename.
        delimiter (str, optional): The delimiter to use for CSV files. Defaults to ','.
        encoding (str, optional): The encoding to use for text-based files. Defaults to 'utf-8'.

    Returns:
        tuple: A tuple containing the loaded Pandas DataFrame, an error message (if any),
               and a value for the temporary file path (always None as it's not used).
    """
    df = None
    error = None
    source_to_read = source_path_or_file

    try:
        # Infer file type if not provided
        if not file_type and isinstance(source_path_or_file, str):
            file_type = _get_file_extension(source_path_or_file)

        # Standardize file type string
        if file_type and file_type.startswith('.'):
            file_type = file_type[1:]

        # For URL, get raw content in memory
        if source_type == 'url':
            url = _convert_gdrive_url(source_path_or_file)
            url = _convert_github_url(url)
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            
            # Use BytesIO for binary files like Excel/Parquet, and StringIO for text
            if file_type in ['xls', 'xlsx', 'parquet']:
                 source_to_read = io.BytesIO(response.content)
            else: # csv, json, and other text-based formats
                 source_to_read = io.StringIO(response.content.decode(encoding))

        # Read the data based on file type
        if file_type == 'csv':
            df = pd.read_csv(source_to_read, delimiter=delimiter, encoding=encoding)
        elif file_type in ['xls', 'xlsx']:
            df = pd.read_excel(source_to_read)
        elif file_type == 'json':
            df = pd.read_json(source_to_read, encoding=encoding)
        elif file_type == 'parquet':
            df = pd.read_parquet(source_to_read)
        else:
            # As a fallback, try reading as CSV
            try:
                if hasattr(source_to_read, 'seek'):
                    source_to_read.seek(0)
                df = pd.read_csv(source_to_read, delimiter=delimiter, encoding=encoding)
                error = f"Warning: File type '{file_type}' not explicitly supported. Attempted to read as CSV."
            except Exception as fallback_e:
                error = f"Error: Unsupported file type '{file_type}'. Could not read as CSV either. Details: {fallback_e}"

    except requests.exceptions.RequestException as e:
        error = f"Error accessing URL: {e}"
    except pd.errors.ParserError as e:
        error = f"Error parsing data. Please check the file format and parameters like delimiter. Details: {e}"
    except FileNotFoundError:
        error = f"Error: File not found at {source_path_or_file}."
    except Exception as e:
        error = f"An unexpected error occurred during data ingestion: {e}"

    # Return None for temp_file_path to maintain function signature compatibility with app.py
    return df, error, None