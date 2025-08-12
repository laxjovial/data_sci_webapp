# app.py

import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from utils.data_ingestion import load_data, get_dataframe_summary
from utils.data_cleaning import (
    handle_missing_values, rename_column, convert_dtype,
    remove_duplicates, standardize_text, handle_outliers, correct_inconsistencies
)
from utils.data_engineering import (
    create_new_feature, apply_encoding, scale_features, rename_and_drop_columns
)
from utils.eda import generate_univariate_plot, generate_bivariate_plot, generate_multivariate_plot
import pandas as pd
import numpy as np
import io

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

PIPELINE_STAGES = [
    "Data Ingestion", "Data Cleaning", "EDA & Visualization",
    "Feature Engineering", "Model Building", "Model Evaluation", "Export & Finalization"
]

def _get_df_from_filepath():
    """
    Retrieves the DataFrame from the file path stored in the session.
    """
    if 'filepath' in session:
        filepath = session['filepath']
        file_ext = session.get('file_ext', '.csv') # Default to csv
        try:
            if file_ext == '.csv':
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            return df, None
        except Exception as e:
            return None, f"Error reading file: {e}"
    return None, "No dataset loaded. Please upload one."

def _save_df_to_filepath(df):
    """
    Saves the DataFrame to the file path stored in the session.
    """
    if 'filepath' in session:
        filepath = session['filepath']
        file_ext = session.get('file_ext', '.csv')
        try:
            if file_ext == '.csv':
                df.to_csv(filepath, index=False)
            else:
                df.to_excel(filepath, index=False)
            return True, None
        except Exception as e:
            return False, f"Error saving file: {e}"
    return False, "Filepath not found in session."

def _get_progress_data(current_stage_name):
    try:
        current_index = PIPELINE_STAGES.index(current_stage_name)
        progress_percent = int(((current_index + 1) / len(PIPELINE_STAGES)) * 100)
    except ValueError:
        current_index = 0
        progress_percent = 0
    return current_stage_name, progress_percent


@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    if request.method == 'POST':
        source_type = request.form.get('source_type')
        df = None

        try:
            if source_type == 'url':
                url = request.form.get('url')
                df, error_message = load_data(source_type='url', source_path_or_file=url)
                if df is not None:
                    file_ext = '.csv' if '.csv' in url else '.xlsx'
            elif source_type == 'upload':
                file = request.files.get('file')
                if file and file.filename:
                    _, file_ext = os.path.splitext(file.filename)
                    if file_ext.lower() not in ['.csv', '.xls', '.xlsx']:
                        raise ValueError("Unsupported file type. Please upload CSV or Excel.")
                    df, error_message = load_data(source_type='upload', source_path_or_file=file)
                else:
                    raise ValueError("No file selected for upload.")
            else:
                raise ValueError("Invalid data source type.")

            if df is not None:
                filename = f"{uuid.uuid4()}{file_ext}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                session['filepath'] = filepath
                session['file_ext'] = file_ext
                _save_df_to_filepath(df)
                return redirect(url_for('data_viewer'))

        except Exception as e:
            error_message = str(e)

    current_stage, progress_percent = _get_progress_data("Data Ingestion")
    return render_template('index.html', error=error_message, current_stage=current_stage, progress_percent=progress_percent)


@app.route('/data_viewer')
def data_viewer():
    df, error_message = _get_df_from_filepath()
    if error_message:
        return redirect(url_for('index'))

    df_head_html, info_str, desc_html, columns, unique_values = get_dataframe_summary(df)
    current_stage, progress_percent = _get_progress_data("Data Cleaning")
    return render_template('data_viewer.html',
                           df_head=df_head_html, df_info=info_str, df_desc=desc_html,
                           unique_values=unique_values, columns=columns,
                           current_stage=current_stage, progress_percent=progress_percent)


@app.route('/data_cleaning', methods=['GET', 'POST'])
def data_cleaning():
    df, error_message = _get_df_from_filepath()
    if error_message:
        return redirect(url_for('index'))

    new_error_message = None
    success_message = None

    if request.method == 'POST':
        action = request.form.get('action')
        
        try:
            if action == 'handle_missing':
                columns = request.form.getlist('missing_cols')
                strategy = request.form.get('missing_strategy')
                df = handle_missing_values(df, columns, strategy)
                success_message = f"Missing values handled using '{strategy}'."

            elif action == 'convert_dtype':
                column = request.form.get('dtype_col')
                dtype = request.form.get('new_dtype')
                df, new_error_message = convert_dtype(df, column, dtype)
                if new_error_message is None:
                    success_message = f"Column '{column}' converted to '{dtype}'."

            elif action == 'remove_duplicates':
                df = remove_duplicates(df)
                success_message = "Duplicate rows removed."
            
            elif action == 'standardize_text':
                column = request.form.get('text_col')
                case = request.form.get('text_case')
                df, new_error_message = standardize_text(df, column, case)
                if new_error_message is None:
                    success_message = f"Text in '{column}' standardized to '{case}'."

            elif action == 'handle_outliers':
                column = request.form.get('outlier_col')
                method = request.form.get('outlier_method')
                df, new_error_message = handle_outliers(df, column, method)
                if new_error_message is None:
                    success_message = f"Outliers in '{column}' handled using '{method}'."

            elif action == 'correct_inconsistencies':
                column = request.form.get('inconsistency_col')
                mapping_str = request.form.get('mapping_dict')
                mapping_dict = eval(mapping_str)
                if not isinstance(mapping_dict, dict):
                    raise ValueError("Input is not a valid dictionary.")
                df, new_error_message = correct_inconsistencies(df, column, mapping_dict)
                if new_error_message is None:
                    success_message = f"Inconsistencies in '{column}' corrected."

            if new_error_message is None:
                _save_df_to_filepath(df)

        except Exception as e:
            new_error_message = f"An error occurred: {e}"

    df_head_html, _, _, columns, _ = get_dataframe_summary(df)
    current_stage, progress_percent = _get_progress_data("Data Cleaning")
    return render_template('data_cleaning.html',
                           df_head=df_head_html, columns=columns,
                           error=new_error_message, success=success_message,
                           current_stage=current_stage, progress_percent=progress_percent)


@app.route('/data_eda', methods=['GET', 'POST'])
def data_eda():
    df, error_message = _get_df_from_filepath()
    if error_message:
        return redirect(url_for('index'))

    df_head_html, info_str, desc_html, columns, unique_values = get_dataframe_summary(df)
    plot_json = None
    new_error_message = None

    if request.method == 'POST':
        try:
            analysis_type = request.form.get('analysis_type')

            if analysis_type == 'univariate':
                column = request.form.get('uni_column')
                if column:
                    plot_json = generate_univariate_plot(df, column)
                else:
                    new_error_message = "Please select a column for univariate analysis."

            elif analysis_type == 'bivariate':
                x_col = request.form.get('bi_x_column')
                y_col = request.form.get('bi_y_column')
                if x_col and y_col:
                    plot_json = generate_bivariate_plot(df, x_col, y_col)
                else:
                    new_error_message = "Please select two columns for bivariate analysis."

            elif analysis_type == 'multivariate':
                multi_columns = request.form.getlist('multi_columns')
                if len(multi_columns) > 1:
                    plot_json = generate_multivariate_plot(df, multi_columns)
                else:
                    new_error_message = "Please select at least two columns for multivariate analysis."

        except Exception as e:
            new_error_message = f"An error occurred during plot generation: {e}"


    current_stage, progress_percent = _get_progress_data("EDA & Visualization")
    return render_template('data_eda.html',
                           df_head=df_head_html, df_info=info_str, df_desc=desc_html,
                           columns=columns, plot_json=plot_json, error=new_error_message,
                           unique_values=unique_values, current_stage=current_stage,
                           progress_percent=progress_percent)


@app.route('/data_engineering', methods=['GET', 'POST'])
def data_engineering():
    df, error_message = _get_df_from_filepath()
    if error_message:
        return redirect(url_for('index'))

    new_error_message = None
    success_message = None
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        try:
            if action == 'create_feature':
                col1 = request.form.get('col1')
                col2 = request.form.get('col2')
                operation = request.form.get('operation')
                new_col_name = request.form.get('new_col_name')
                df, new_error_message = create_new_feature(df, col1, col2, operation, new_col_name)
                if new_error_message is None:
                    success_message = f"New feature '{new_col_name}' created."

            elif action == 'apply_encoding':
                columns = request.form.getlist('columns')
                encoding_type = request.form.get('encoding_type')
                df, new_error_message = apply_encoding(df, columns, encoding_type)
                if new_error_message is None:
                    success_message = f"Columns encoded using '{encoding_type}'."

            elif action == 'scale_features':
                columns = request.form.getlist('columns')
                scaler_type = request.form.get('scaler_type')
                df, new_error_message = scale_features(df, columns, scaler_type)
                if new_error_message is None:
                    success_message = f"Columns scaled using '{scaler_type}'."

            elif action == 'rename_drop':
                old_col_name = request.form.get('old_column_name')
                new_col_name = request.form.get('new_column_name')
                columns_to_drop = request.form.getlist('columns_to_drop')
                df, new_error_message = rename_and_drop_columns(df, new_column_name, old_col_name, columns_to_drop)
                if new_error_message is None:
                    success_message = "Columns renamed/dropped."

            if new_error_message is None:
                _save_df_to_filepath(df)

        except Exception as e:
            new_error_message = f"An error occurred: {e}"

    df_head_html, _, _, columns, _ = get_dataframe_summary(df)
    current_stage, progress_percent = _get_progress_data("Feature Engineering")
    return render_template('data_engineering.html',
                           df_head=df_head_html, columns=columns, error=new_error_message,
                           success=success_message, current_stage=current_stage,
                           progress_percent=progress_percent)


@app.route('/user_guide')
def user_guide():
    """
    Renders the user guide page.
    """
    return render_template('user_guide.html')


if __name__ == '__main__':
    app.run(debug=True)
