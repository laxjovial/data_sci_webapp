# app.py

from flask import Flask, render_template, request, redirect, url_for, session
from utils.data_ingestion import load_data, get_dataframe_summary
from utils.data_cleaning import (
    handle_missing_values, rename_column, convert_dtype, 
    remove_duplicates, standardize_text, handle_outliers, correct_inconsistencies
)
from utils.data_engineering import (
    create_new_feature, apply_encoding, scale_features, rename_and_drop_columns
)
from utils.data_visualization import generate_plot
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

    Args:
        current_stage_name (str): The name of the current stage.

    Returns:
        tuple: The current stage name and the progress percentage.
    """
    try:
        current_index = PIPELINE_STAGES.index(current_stage_name)
        progress_percent = int(((current_index + 1) / len(PIPELINE_STAGES)) * 100)
    except ValueError:
        current_index = 0
        progress_percent = 0
    return current_stage_name, progress_percent

def _save_df_to_session(df):
    """
    Helper function to safely save the DataFrame to the session.
    """
    session['df'] = df.to_json()


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the data ingestion stage, allowing users to upload a file or
    provide a URL.
    """
    error_message = None
    if request.method == 'POST':
        source_type = request.form.get('source_type')
        
        if source_type == 'url':
            url = request.form.get('url')
            df, error_message = load_data(url=url)
        elif source_type == 'upload':
            file = request.files.get('file')
            df, error_message = load_data(file=file)
        else:
            df = None
            error_message = "Invalid data source type."

        if df is not None:
            _save_df_to_session(df)
            return redirect(url_for('data_viewer'))

    current_stage, progress_percent = _get_progress_data("Data Ingestion")
    return render_template('index.html', error=error_message, current_stage=current_stage, progress_percent=progress_percent)

@app.route('/data_viewer')
def data_viewer():
    """
    Displays a summary of the loaded DataFrame.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        return redirect(url_for('index'))

    df_head_html, info_str, desc_html, columns, unique_values = get_dataframe_summary(df)

    current_stage, progress_percent = _get_progress_data("Data Cleaning")
    
    return render_template('data_viewer.html',
                           df_head=df_head_html,
                           df_info=info_str,
                           df_desc=desc_html,
                           unique_values=unique_values,
                           columns=columns,
                           current_stage=current_stage,
                           progress_percent=progress_percent)


@app.route('/data_cleaning', methods=['GET', 'POST'])
def data_cleaning():
    """
    Renders the Data Cleaning page and handles data cleaning operations.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        return redirect(url_for('index'))

    new_error_message = None
    success_message = None

    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'handle_missing':
            columns = request.form.getlist('missing_cols')
            strategy = request.form.get('missing_strategy')
            df_cleaned = handle_missing_values(df, columns, strategy)
            if df_cleaned is not None:
                _save_df_to_session(df_cleaned)
                df = df_cleaned
                success_message = f"Missing values handled successfully using '{strategy}' strategy."
            else:
                new_error_message = "Failed to handle missing values. Please check your selections."

        elif action == 'convert_dtype':
            column = request.form.get('dtype_col')
            dtype = request.form.get('new_dtype')
            df_cleaned, new_error_message = convert_dtype(df, column, dtype)
            if df_cleaned is not None:
                _save_df_to_session(df_cleaned)
                df = df_cleaned
                success_message = f"Column '{column}' converted to '{dtype}' successfully."
            
        elif action == 'remove_duplicates':
            df_cleaned = remove_duplicates(df)
            _save_df_to_session(df_cleaned)
            df = df_cleaned
            success_message = "Duplicate rows removed successfully."
        
        elif action == 'standardize_text':
            column = request.form.get('text_col')
            case = request.form.get('text_case')
            df_cleaned, new_error_message = standardize_text(df, column, case)
            if df_cleaned is not None:
                _save_df_to_session(df_cleaned)
                df = df_cleaned
                success_message = f"Text in column '{column}' standardized to '{case}' case successfully."

        elif action == 'handle_outliers':
            column = request.form.get('outlier_col')
            method = request.form.get('outlier_method')
            df_cleaned, new_error_message = handle_outliers(df, column, method)
            if df_cleaned is not None:
                _save_df_to_session(df_cleaned)
                df = df_cleaned
                success_message = f"Outliers in column '{column}' handled successfully using '{method}' method."
        
        elif action == 'correct_inconsistencies':
            column = request.form.get('inconsistency_col')
            mapping_str = request.form.get('mapping_dict')
            try:
                mapping_dict = eval(mapping_str)
                if not isinstance(mapping_dict, dict):
                    raise ValueError("Input is not a valid dictionary.")
                df_cleaned, new_error_message = correct_inconsistencies(df, column, mapping_dict)
                if df_cleaned is not None:
                    _save_df_to_session(df_cleaned)
                    df = df_cleaned
                    success_message = f"Inconsistencies in column '{column}' corrected successfully."
            except Exception as e:
                new_error_message = f"Error with mapping dictionary: {e}"
        
        df_head_html, _, _, columns, _ = get_dataframe_summary(df)
        
    else:
        df_head_html, _, _, columns, _ = get_dataframe_summary(df)

    current_stage, progress_percent = _get_progress_data("Data Cleaning")

    return render_template('data_cleaning.html',
                           df_head=df_head_html,
                           columns=columns,
                           error=new_error_message,
                           success=success_message,
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
                           unique_values=unique_values,
                           current_stage=current_stage,
                           progress_percent=progress_percent)


@app.route('/data_engineering', methods=['GET', 'POST'])
def data_engineering():
    """
    Renders the Feature Engineering page and handles operations.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        return redirect(url_for('index'))

    new_error_message = None
    success_message = None
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'create_feature':
            col1 = request.form.get('col1')
            col2 = request.form.get('col2')
            operation = request.form.get('operation')
            new_col_name = request.form.get('new_col_name')
            df_engineered, new_error_message = create_new_feature(df, col1, col2, operation, new_col_name)
            if df_engineered is not None:
                _save_df_to_session(df_engineered)
                df = df_engineered
                success_message = f"New feature '{new_col_name}' created successfully."

        elif action == 'apply_encoding':
            columns = request.form.getlist('columns')
            encoding_type = request.form.get('encoding_type')
            df_engineered, new_error_message = apply_encoding(df, columns, encoding_type)
            if df_engineered is not None:
                _save_df_to_session(df_engineered)
                df = df_engineered
                success_message = f"Columns {columns} encoded using '{encoding_type}' successfully."

        elif action == 'scale_features':
            columns = request.form.getlist('columns')
            scaler_type = request.form.get('scaler_type')
            df_engineered, new_error_message = scale_features(df, columns, scaler_type)
            if df_engineered is not None:
                _save_df_to_session(df_engineered)
                df = df_engineered
                success_message = f"Columns {columns} scaled using '{scaler_type}' successfully."

        elif action == 'rename_drop':
            old_col_name = request.form.get('old_column_name')
            new_col_name = request.form.get('new_column_name')
            columns_to_drop = request.form.getlist('columns_to_drop')
            df_engineered, new_error_message = rename_and_drop_columns(df, new_column_name, old_col_name, columns_to_drop)
            if df_engineered is not None:
                _save_df_to_session(df_engineered)
                df = df_engineered
                success_message = "Columns renamed/dropped successfully."
        
        df_head_html, _, _, columns, _ = get_dataframe_summary(df)

    else:
        df_head_html, _, _, columns, _ = get_dataframe_summary(df)

    current_stage, progress_percent = _get_progress_data("Feature Engineering")

    return render_template('data_engineering.html',
                           df_head=df_head_html,
                           columns=columns,
                           error=new_error_message,
                           success=success_message,
                           current_stage=current_stage,
                           progress_percent=progress_percent)


if __name__ == '__main__':
    app.run(debug=True)
