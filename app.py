# app.py

from flask import Flask, render_template, request, redirect, url_for, session
from utils.data_ingestion import load_data, get_dataframe_summary
from utils.data_cleaning import (
    handle_missing_values, rename_column, convert_dtype, 
    remove_duplicates, standardize_text, handle_outliers, correct_inconsistencies
)
from utils.data_engineering import apply_encoding, remove_outliers
import pandas as pd
import numpy as np
import io

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

    Returns:
        tuple: A tuple containing the current stage name and progress percentage.
    """
    try:
        current_index = PIPELINE_STAGES.index(current_stage_name)
        progress_percent = int((current_index / (len(PIPELINE_STAGES) - 1)) * 100)
    except ValueError:
        progress_percent = 0
    return current_stage_name, progress_percent

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the data ingestion phase.
    """
    error_message = None
    if request.method == 'POST':
        source_type = request.form.get('source_type')
        if source_type == 'url':
            url = request.form.get('url')
            if url:
                df, error_message = load_data(url=url)
                if df is not None:
                    session['df'] = df.to_json()
                    return redirect(url_for('data_viewer'))
            else:
                error_message = "Please provide a valid URL."
        elif source_type == 'upload':
            file = request.files.get('file')
            if file:
                df, error_message = load_data(file=file)
                if df is not None:
                    session['df'] = df.to_json()
                    return redirect(url_for('data_viewer'))
            else:
                error_message = "Please select a file to upload."
    
    current_stage, progress_percent = _get_progress_data("Data Ingestion")
    
    return render_template('index.html', 
                           error=error_message, 
                           current_stage=current_stage, 
                           progress_percent=progress_percent)

@app.route('/data_viewer')
def data_viewer():
    """
    Renders the data viewing page with a summary of the loaded DataFrame.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        return redirect(url_for('index'))

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

@app.route('/data_cleaning', methods=['GET', 'POST'])
def data_cleaning():
    """
    Renders the data cleaning page and handles cleaning operations.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        return redirect(url_for('index'))

    new_error_message = None

    if request.method == 'POST':
        action = request.form.get('action')
        columns_to_process = request.form.getlist('columns')
        
        if not columns_to_process:
            new_error_message = "Please select at least one column."
        else:
            if action == 'missing_values':
                strategy = request.form.get('strategy')
                df = handle_missing_values(df, columns_to_process, strategy)
            elif action == 'rename_column':
                old_col = request.form.get('old_col')
                new_col = request.form.get('new_col')
                if old_col and new_col:
                    df = rename_column(df, old_col, new_col)
                else:
                    new_error_message = "Please provide both old and new column names."
            elif action == 'convert_dtype':
                dtype_col = request.form.get('dtype_col')
                new_dtype = request.form.get('new_dtype')
                if dtype_col and new_dtype:
                    df, new_error_message = convert_dtype(df, dtype_col, new_dtype)
                else:
                    new_error_message = "Please select a column and a data type."
            elif action == 'remove_duplicates':
                df = remove_duplicates(df, columns_to_process)
            elif action == 'standardize_text':
                standardize_col = request.form.get('standardize_col')
                if standardize_col:
                    df = standardize_text(df, standardize_col)
                else:
                    new_error_message = "Please select a column to standardize."
            elif action == 'handle_outliers':
                outlier_col = request.form.get('outlier_col')
                outlier_method = request.form.get('outlier_method')
                if outlier_col and outlier_method:
                    df, new_error_message = handle_outliers(df, outlier_col, outlier_method)
                else:
                    new_error_message = "Please select a column and a method."
            elif action == 'correct_inconsistencies':
                inconsist_col = request.form.get('inconsist_col')
                mapping_str = request.form.get('mapping_dict')
                try:
                    mapping_dict = eval(mapping_str)
                    df, new_error_message = correct_inconsistencies(df, inconsist_col, mapping_dict)
                except Exception as e:
                    new_error_message = f"Invalid dictionary format: {e}"

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

@app.route('/feature_engineering', methods=['GET', 'POST'])
def feature_engineering():
    """
    Renders the feature engineering page and handles encoding operations.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        return redirect(url_for('index'))
    
    new_error_message = None
    
    if request.method == 'POST':
        encoding_type = request.form.get('encoding_type')
        columns_to_encode = request.form.getlist('columns')
        
        if not columns_to_encode:
            new_error_message = "Please select at least one column to encode."
        else:
            df, new_error_message = apply_encoding(df, columns_to_encode, encoding_type)
            session['df'] = df.to_json()
    
    df_head_html, info_str, desc_html, columns, unique_values = get_dataframe_summary(df)

    current_stage, progress_percent = _get_progress_data("Feature Engineering")

    return render_template('feature_engineering.html',
                           df_head=df_head_html,
                           columns=columns,
                           error=new_error_message,
                           current_stage=current_stage,
                           progress_percent=progress_percent)

@app.route('/model_building')
def model_building():
    """
    Renders a placeholder for the model building stage.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        return redirect(url_for('index'))
    
    df_head_html, _, _, _, _ = get_dataframe_summary(df)
    
    current_stage, progress_percent = _get_progress_data("Model Building")

    return render_template('model_building.html',
                           df_head=df_head_html,
                           current_stage=current_stage,
                           progress_percent=progress_percent)


if __name__ == '__main__':
    app.run(debug=True)
