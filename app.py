# app.py

from flask import Flask, render_template, request, redirect, url_for, session
from utils.data_ingestion import load_data, get_dataframe_summary
from utils.data_cleaning import (
    handle_missing_values, rename_column, convert_dtype, 
    remove_duplicates, standardize_text, handle_outliers, correct_inconsistencies
)
from utils.data_visualization import generate_plot
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here' # For session management

# Define the stages of our data science pipeline
PIPELINE_STAGES = [
    "Data Ingestion",
    "Data Cleaning",
    "EDA & Visualization",
    "Feature Engineering",
    "Model Building",
    "Model Evaluation",
    "Export & Finalization"
]

def _get_df_from_session():
    """
    Helper function to safely retrieve the DataFrame from the session.
    
    Returns:
        pd.DataFrame: The DataFrame from the session.
        str: An error message if the DataFrame is not found, otherwise None.
    """
    if 'df' in session:
        try:
            df = pd.read_json(session['df'])
            return df, None
        except Exception as e:
            return None, f"Error restoring DataFrame from session: {e}"
    return None, "No dataset loaded. Please go back to the home page to upload one."

def _get_progress_data(current_stage_name):
    """
    Calculates the current progress based on the pipeline stages.

    Technical: This function finds the index of the current stage in the
    PIPELINE_STAGES list and calculates the percentage completion. It returns
    the current stage name and the percentage to be used in the template.

    Layman: This function simply looks at what you're doing right now (e.g.,
    Data Cleaning), figures out where that is on our master to-do list, and
    calculates how far along you are in the project.

    Args:
        current_stage_name (str): The name of the current stage.

    Returns:
        tuple: (current_stage_name, progress_percentage)
    """
    try:
        current_index = PIPELINE_STAGES.index(current_stage_name)
        progress_percent = int(((current_index + 1) / len(PIPELINE_STAGES)) * 100)
        return current_stage_name, progress_percent
    except ValueError:
        return "Unknown Stage", 0


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    The main route for the web application, now with file upload functionality.
    
    Handles both GET and POST requests. The GET request renders the initial
    home page with options to upload data from URL or a local file. The POST 
    request processes the submitted data source and redirects to the data 
    viewing page upon success. It now checks for a file in the request first.
    """
    if request.method == 'POST':
        source_type = request.form.get('source_type')
        
        # New logic to handle file uploads
        if 'file' in request.files and request.files['file'].filename != '':
            source_file = request.files['file']
            df, error_message = load_data('upload', source_file)
        
        # Existing logic to handle URLs
        elif 'source_path' in request.form and request.form.get('source_path') != '':
            source_path = request.form.get('source_path')
            df, error_message = load_data('url', source_path)
            
        else:
            current_stage, progress_percent = _get_progress_data("Data Ingestion")
            return render_template('index.html', error="Please provide a valid file or URL.", current_stage=current_stage, progress_percent=progress_percent)

        if error_message:
            current_stage, progress_percent = _get_progress_data("Data Ingestion")
            return render_template('index.html', error=error_message, current_stage=current_stage, progress_percent=progress_percent)

        session['df'] = df.to_json()
        return redirect(url_for('data_viewer'))

    current_stage, progress_percent = _get_progress_data("Data Ingestion")
    return render_template('index.html', current_stage=current_stage, progress_percent=progress_percent)


@app.route('/data_viewer')
def data_viewer():
    """
    Displays the loaded dataset to the user.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        current_stage, progress_percent = _get_progress_data("Data Ingestion")
        return render_template('index.html', error=error_message, current_stage=current_stage, progress_percent=progress_percent)

    df_head_html, info_str, desc_html, columns, unique_values = get_dataframe_summary(df)

    current_stage, progress_percent = _get_progress_data("Data Ingestion")
    
    return render_template('data_viewer.html', 
                           df_head=df_head_html,
                           df_info=info_str,
                           df_desc=desc_html,
                           columns=columns,
                           unique_values=unique_values,
                           current_stage=current_stage,
                           progress_percent=progress_percent)

@app.route('/data_cleaning')
def data_cleaning():
    """
    Renders the data cleaning page.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        current_stage, progress_percent = _get_progress_data("Data Ingestion")
        return redirect(url_for('index'))

    df_head_html, _, _, columns, _ = get_dataframe_summary(df)
    
    current_stage, progress_percent = _get_progress_data("Data Cleaning")

    return render_template('data_cleaning.html',
                           df_head=df_head_html,
                           columns=columns,
                           current_stage=current_stage,
                           progress_percent=progress_percent)


@app.route('/clean_data', methods=['POST'])
def clean_data():
    """
    Processes the data cleaning requests from the form.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        return redirect(url_for('index'))

    action_type = request.form.get('action_type')
    new_error_message = None

    if action_type == 'handle_missing':
        columns = request.form.getlist('columns')
        strategy = request.form.get('strategy')
        if columns and strategy:
            df = handle_missing_values(df, columns, strategy)
    
    elif action_type == 'rename_column':
        old_col = request.form.get('old_col')
        new_col = request.form.get('new_col')
        if old_col and new_col:
            df = rename_column(df, old_col, new_col)

    elif action_type == 'convert_type':
        column = request.form.get('col_to_convert')
        new_type = request.form.get('new_type')
        if column and new_type:
            df, new_error_message = convert_dtype(df, column, new_type)

    elif action_type == 'remove_duplicates':
        df = remove_duplicates(df)
        new_error_message = "Duplicate rows have been removed."

    elif action_type == 'standardize_text':
        columns = request.form.getlist('standardize_cols')
        if columns:
            df = standardize_text(df, columns)
    
    elif action_type == 'handle_outliers':
        column = request.form.get('outlier_col')
        method = request.form.get('outlier_method')
        if column and method:
            df, new_error_message = handle_outliers(df, column, method)

    session['df'] = df.to_json()
    
    df_head_html, _, _, columns, _ = get_dataframe_summary(df)
    current_stage, progress_percent = _get_progress_data("Data Cleaning")

    return render_template('data_cleaning.html',
                           df_head=df_head_html,
                           columns=columns,
                           error=new_error_message,
                           current_stage=current_stage,
                           progress_percent=progress_percent)

@app.route('/data_eda', methods=['GET', 'POST'])
def data_eda():
    """
    Renders the EDA and Visualization page.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        return redirect(url_for('index'))

    df_head_html, info_str, desc_html, columns, unique_values = get_dataframe_summary(df)

    plot_json = None
    if request.method == 'POST':
        plot_type = request.form.get('plot_type')
        x_col = request.form.get('x_col')
        y_col = request.form.get('y_col')

        plot_json, error_message = generate_plot(df, plot_type, x_col, y_col)

    current_stage, progress_percent = _get_progress_data("EDA & Visualization")

    return render_template('data_eda.html',
                           df_head=df_head_html,
                           df_info=info_str,
                           df_desc=desc_html,
                           columns=columns,
                           plot_json=plot_json,
                           error=error_message,
                           current_stage=current_stage,
                           progress_percent=progress_percent)

if __name__ == '__main__':
    app.run(debug=True)
