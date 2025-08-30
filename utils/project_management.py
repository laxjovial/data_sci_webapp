import os
import json
import shutil
import pandas as pd
from datetime import datetime

PROJECTS_DIR = 'projects/'

def _ensure_projects_dir():
    """Ensures the main projects directory exists."""
    os.makedirs(PROJECTS_DIR, exist_ok=True)

def save_project(project_name, df, history):
    """
    Saves the current state of a project, including the Pandas DataFrame.

    Technical: A new directory is created for the project. The Pandas DataFrame is
    saved as a single Parquet file, which is efficient for storage. Project
    metadata and the operation history are saved as a JSON file.

    Layman: This function saves all your work on a project. It creates a special
    folder for your project and stores your data file and a log of all the steps
    you've taken so far.

    Args:
        project_name (str): The name for the project.
        df (pd.DataFrame): The current Pandas DataFrame to save.
        history (list): A list of operations performed.

    Returns:
        tuple: (success, message)
    """
    _ensure_projects_dir()
    project_path = os.path.join(PROJECTS_DIR, project_name)
    if os.path.exists(project_path):
        return False, "A project with this name already exists."

    try:
        os.makedirs(project_path)
        # Save the Pandas DataFrame to a single Parquet file
        df.to_parquet(os.path.join(project_path, 'data.parquet'))
        # Save metadata and history
        metadata = {
            'name': project_name,
            'created_at': datetime.now().isoformat(),
            'history': history
        }
        with open(os.path.join(project_path, 'project.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        return True, "Project saved successfully."
    except Exception as e:
        # Clean up the directory if an error occurs during saving
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
        return False, f"Error saving project: {e}"

def load_project(project_name):
    """
    Loads a project's Pandas DataFrame and history.

    Technical: The function looks for a project directory by name. It then uses
    pandas' `read_parquet` to load the data from the 'data.parquet' file. The
    metadata, including the operation history, is loaded from the 'project.json' file.

    Layman: This function opens a previously saved project. It gets the data you were
    working on and all the steps you performed, so you can continue right where you left off.

    Args:
        project_name (str): The name of the project to load.

    Returns:
        tuple: (Pandas DataFrame, history, error_message)
    """
    project_path = os.path.join(PROJECTS_DIR, project_name)
    if not os.path.exists(project_path):
        return None, None, "Project not found."

    try:
        df = pd.read_parquet(os.path.join(project_path, 'data.parquet'))
        with open(os.path.join(project_path, 'project.json'), 'r') as f:
            metadata = json.load(f)
        history = metadata.get('history', [])
        return df, history, None
    except Exception as e:
        return None, None, f"Error loading project: {e}"

def list_projects():
    """
    Lists all saved projects.

    Technical: The function scans the main projects directory. For each subdirectory,
    it attempts to read the 'project.json' file to retrieve metadata. Invalid directories
    (those without a 'project.json') are skipped. The list is sorted by creation date.

    Layman: This shows you a list of all the projects you've saved, along with
    their details like when they were created.

    Returns:
        list: A list of project metadata dictionaries.
    """
    _ensure_projects_dir()
    projects = []
    for project_name in os.listdir(PROJECTS_DIR):
        project_path = os.path.join(PROJECTS_DIR, project_name)
        if os.path.isdir(project_path):
            try:
                with open(os.path.join(project_path, 'project.json'), 'r') as f:
                    metadata = json.load(f)
                    projects.append(metadata)
            except (FileNotFoundError, json.JSONDecodeError):
                continue  # Skip directories that are not valid projects
    return sorted(projects, key=lambda x: x.get('created_at', ''), reverse=True)

def delete_project(project_name):
    """
    Deletes a project.

    Technical: The function removes the entire project directory and all its contents,
    including the data and metadata files, using `shutil.rmtree`. This provides a
    complete and clean deletion.

    Layman: This permanently deletes a project and all its data.

    Args:
        project_name (str): The name of the project to delete.

    Returns:
        tuple: (success, message)
    """
    project_path = os.path.join(PROJECTS_DIR, project_name)
    if not os.path.exists(project_path):
        return False, "Project not found."

    try:
        shutil.rmtree(project_path)
        return True, "Project deleted successfully."
    except Exception as e:
        return False, f"Error deleting project: {e}"