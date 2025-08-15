# utils/data_ingestion.py
import dask.dataframe as dd
import pandas as pd
import requests
import io
import re
import os
import tempfile

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

def load_data(source_path_or_file, source_type='upload', file_type=None, delimiter=',', encoding='utf-8'):
    """
    Loads a dataset from a file path or URL as a Dask DataFrame.
    
    Technical: This function first handles URL-based data by safely downloading it
    to a temporary file, which is a robust way to handle remote data streams. It then
    uses Dask's appropriate `read_*` functions to load the data lazily, which is
    essential for large datasets. It includes comprehensive error handling for
    common issues like unsupported file types, network errors, and file not found errors.

    Layman: This function opens your data file, whether it's from your computer or a web link.
    It can automatically figure out if it's a CSV, Excel, JSON, or Parquet file and
    read it correctly, even for very large files that don't fit in your computer's memory.

    Args:
        source_path_or_file (str or file-like object): The local file path, URL, or uploaded file object.
        source_type (str): The source of the data ('upload', 'url').
        file_type (str, optional): The type of file ('csv', 'excel', 'json', 'parquet').
                                   If None, it will be inferred from the filename.
        delimiter (str, optional): The delimiter to use for CSV files. Defaults to ','.
        encoding (str, optional): The encoding to use for text-based files. Defaults to 'utf-8'.

    Returns:
        tuple: A tuple containing the loaded Dask DataFrame, an error message (if any),
               and a temporary file path (if created).
    """
    df = None
    error = None
    temp_file_path = None
    source_to_read = source_path_or_file

    try:
        # Infer file type if not provided
        if not file_type:
            file_type = _get_file_extension(source_path_or_file)

        # Standardize file type string
        if file_type.startswith('.'):
            file_type = file_type[1:]

        # For URL, download to a temporary file
        if source_type == 'url':
            url = _convert_gdrive_url(source_path_or_file)
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
                tmp_file.write(response.content)
                temp_file_path = tmp_file.name
            source_to_read = temp_file_path
        
        # Read the data based on file type
        if file_type == 'csv':
            df = dd.read_csv(source_to_read, delimiter=delimiter, encoding=encoding, blocksize=None)
        elif file_type in ['xls', 'xlsx']:
            # Dask doesn't have a direct excel reader, so we read with pandas and convert
            pandas_df = pd.read_excel(source_to_read)
            df = dd.from_pandas(pandas_df, npartitions=2)
        elif file_type == 'json':
            df = dd.read_json(source_to_read, encoding=encoding, blocksize=None)
        elif file_type == 'parquet':
            df = dd.read_parquet(source_to_read)
        else:
            # As a fallback, try reading as CSV, as it's the most common format
            try:
                df = dd.read_csv(source_to_read, delimiter=delimiter, encoding=encoding, blocksize=None)
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

    # Return the temporary file path along with the DataFrame and error message
    return df, error, temp_file_path