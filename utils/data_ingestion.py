import dask.dataframe as dd
import pandas as pd
import requests
import io
import re
import os

def _get_file_extension(path):
    """Extracts the file extension from a path or URL."""
    try:
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
    """
    df = None
    error = None

    try:
        if not file_type:
            file_type = _get_file_extension(source_path_or_file)

        if file_type.startswith('.'):
            file_type = file_type[1:]

        path_to_read = source_path_or_file
        if source_type == 'url':
            path_to_read = _convert_gdrive_url(source_path_or_file)
        
        if file_type == 'csv':
            df = dd.read_csv(path_to_read, delimiter=delimiter, encoding=encoding, blocksize=None)
        elif file_type in ['xls', 'xlsx']:
            # Dask doesn't directly read excel, so we use pandas and convert
            pandas_df = pd.read_excel(path_to_read)
            df = dd.from_pandas(pandas_df, npartitions=2)
        elif file_type == 'json':
            df = dd.read_json(path_to_read, encoding=encoding, blocksize=None)
        elif file_type == 'parquet':
            df = dd.read_parquet(path_to_read)
        else:
            error = f"Error: Unsupported file type '{file_type}'."

    except Exception as e:
        error = f"An unexpected error occurred during data ingestion: {e}"

    return df, error
