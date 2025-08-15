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
    Saves the current state of a project.

    Args:
        project_name (str): The name for the project.
        df (pd.DataFrame): The current DataFrame to save.
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
        # Save the dataframe
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
        return False, f"Error saving project: {e}"

def load_project(project_name):
    """
    Loads a project's state.

    Args:
        project_name (str): The name of the project to load.

    Returns:
        tuple: (DataFrame, history, error_message)
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
            except FileNotFoundError:
                # Handle cases where a directory might not be a valid project
                continue
    return sorted(projects, key=lambda x: x.get('created_at', ''), reverse=True)

def delete_project(project_name):
    """
    Deletes a project.

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
