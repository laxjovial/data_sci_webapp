# app.py

import os
import uuid

import json
import shutil

from flask import Flask, render_template, request, redirect, url_for, session, send_file
from werkzeug.utils import secure_filename
from utils.data_ingestion import load_data, get_dataframe_summary
from utils.data_cleaning import (
    handle_missing_values, rename_column, convert_dtype,
    remove_duplicates, standardize_text, handle_outliers, correct_inconsistencies,
    format_date_column, sort_dataframe, reset_dataframe_index, drop_columns
)
from utils.data_engineering import (
    create_new_feature, apply_encoding, scale_features, rename_and_drop_columns
)
from utils.data_filtering import filter_dataframe
from utils.data_aggregation import group_by_aggregate, pivot_table

from utils.data_combining import combine_dataframes
from utils.modeling import get_model_list, run_models, get_hyperparameter_grid, tune_model_hyperparameters

from utils.eda import generate_univariate_plot, generate_bivariate_plot, generate_multivariate_plot
from utils.data_export import export_dataframe, export_ipynb
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


def _get_project_dir():
    if 'project_id' in session:
        return os.path.join(app.config['UPLOAD_FOLDER'], session['project_id'])
    return None

def _get_df_from_project():
    project_dir = _get_project_dir()
    if project_dir:
        state_path = os.path.join(project_dir, 'project_state.json')
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            data_path = os.path.join(project_dir, state['data_filename'])
            if state['data_filename'].endswith('.csv'):
                df = pd.read_csv(data_path)
            else:
                df = pd.read_excel(data_path)
            return df, None
        except Exception as e:
            return None, f"Error loading project data: {e}"
    return None, "No project loaded. Please create or load a project."

def _save_df_to_project(df):
    project_dir = _get_project_dir()
    if project_dir:
        state_path = os.path.join(project_dir, 'project_state.json')
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            data_path = os.path.join(project_dir, state['data_filename'])
            if state['data_filename'].endswith('.csv'):
                df.to_csv(data_path, index=False)
            else:
                df.to_excel(data_path, index=False)
            return True, None
        except Exception as e:
            return False, f"Error saving project data: {e}"
    return False, "No project loaded."

def _save_project_state():
    project_dir = _get_project_dir()
    if project_dir:
        state_path = os.path.join(project_dir, 'project_state.json')
        state = {
            'project_id': session.get('project_id'),
            'data_filename': session.get('data_filename'),
            'code_log': session.get('code_log', []),
            'last_run_features': session.get('last_run_features'),
            'last_run_target': session.get('last_run_target'),
            'last_run_problem_type': session.get('last_run_problem_type')
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4)

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

        # This route now only handles creating new projects
        action = request.form.get('action')
        if action != 'create_project':
            return redirect(url_for('index'))

        try:
            source_type = request.form.get('source_type')
            df = None
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
                        raise ValueError("Unsupported file type.")
                    df, error_message = load_data(source_type='upload', source_path_or_file=file)
                else:
                    raise ValueError("No file selected.")

            if df is not None:
                project_id = f"project_{uuid.uuid4().hex[:12]}"
                project_dir = os.path.join(app.config['UPLOAD_FOLDER'], project_id)
                os.makedirs(project_dir, exist_ok=True)

                data_filename = f"data{file_ext}"
                data_path = os.path.join(project_dir, data_filename)

                # Clear previous session data and set up new project
                session.clear()
                session['project_id'] = project_id
                session['data_filename'] = data_filename
                session['code_log'] = [f"df = pd.read_csv('data.csv')"] if file_ext == '.csv' else [f"df = pd.read_excel('data.xlsx')"]

                if file_ext == '.csv':
                    df.to_csv(data_path, index=False)
                else:
                    df.to_excel(data_path, index=False)

                _save_project_state()
                return redirect(url_for('data_viewer'))

        except Exception as e:
            error_message = str(e)

    return render_template('index.html', error=error_message)


@app.route('/projects', methods=['GET', 'POST'])
def projects():
    error_message = None
    success_message = None
    if request.method == 'POST':
        action = request.form.get('action')
        project_id = request.form.get('project_id')
        
        if not project_id or not project_id.startswith('project_'):
            error_message = "Invalid Project ID format."
        else:
            project_dir = os.path.join(app.config['UPLOAD_FOLDER'], project_id)

            if action == 'load_project':
                try:
                    state_path = os.path.join(project_dir, 'project_state.json')
                    if not os.path.exists(state_path):
                        raise ValueError("Project ID not found.")

                    with open(state_path, 'r') as f:
                        state = json.load(f)


                    session.clear()
                    for key, value in state.items():
                        session[key] = value

                    return redirect(url_for('data_viewer'))
                except Exception as e:
                    error_message = f"Error loading project: {e}"

            elif action == 'delete_project':
                try:
                    if os.path.exists(project_dir):
                        shutil.rmtree(project_dir)
                        success_message = f"Project '{project_id}' deleted successfully."
                    else:
                        error_message = "Project not found."
                except Exception as e:
                    error_message = f"Error deleting project: {e}"

    project_list = [d for d in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isdir(os.path.join(app.config['UPLOAD_FOLDER'], d)) and d.startswith('project_')]

    return render_template('projects.html', projects=project_list, error=error_message, success=success_message)



@app.route('/data_viewer')
def data_viewer():
    df, error_message = _get_df_from_project()

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

    df, error_message = _get_df_from_project()

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
                session['code_log'].append(f"df = df.drop_duplicates()")
                success_message = "Duplicate rows removed."
            
            elif action == 'standardize_text':
                column = request.form.get('text_col')
                case = request.form.get('text_case')
                df, new_error_message = standardize_text(df, column, case)
                if new_error_message is None:
                    session['code_log'].append(f"df['{column}'] = df['{column}'].str.{case}()")
                    success_message = f"Text in '{column}' standardized to '{case}'."

            elif action == 'handle_outliers':
                column = request.form.get('outlier_col')
                method = request.form.get('outlier_method')
                df, new_error_message = handle_outliers(df, column, method)
                if new_error_message is None:
                    session['code_log'].append(f"# Handling outliers in '{column}' using {method} method (code is complex)")
                    success_message = f"Outliers in '{column}' handled using '{method}'."

            elif action == 'correct_inconsistencies':
                column = request.form.get('inconsistency_col')
                mapping_str = request.form.get('mapping_dict')
                mapping_dict = eval(mapping_str)
                if not isinstance(mapping_dict, dict):
                    raise ValueError("Input is not a valid dictionary.")
                df, new_error_message = correct_inconsistencies(df, column, mapping_dict)
                if new_error_message is None:
                    session['code_log'].append(f"df['{column}'] = df['{column}'].replace({mapping_dict})")
                    success_message = f"Inconsistencies in '{column}' corrected."

            elif action == 'format_dates':
                column = request.form.get('date_col')
                date_format = request.form.get('date_format') or None # Use None if empty string
                df, new_error_message = format_date_column(df, column, date_format)
                if new_error_message is None:
                    session['code_log'].append(f"df['{column}'] = pd.to_datetime(df['{column}'], format='{date_format}', errors='coerce')")
                    success_message = f"Date column '{column}' formatted successfully."

            elif action == 'sort_data':
                columns = request.form.getlist('sort_cols')
                # The form sends 'True' or 'False' as strings
                ascending = request.form.get('sort_order') == 'True'
                # Create a list of booleans of the same length as columns
                ascending_list = [ascending] * len(columns)
                df, new_error_message = sort_dataframe(df, columns, ascending_list)
                if new_error_message is None:
                    session['code_log'].append(f"df = df.sort_values(by={columns}, ascending={ascending})")
                    success_message = f"Data sorted by {columns}."

            elif action == 'drop_columns':
                columns = request.form.getlist('drop_cols')
                df, new_error_message = drop_columns(df, columns)
                if new_error_message is None:
                    session['code_log'].append(f"df = df.drop(columns={columns})")
                    success_message = f"Columns {columns} dropped."

            elif action == 'reset_index':
                df, new_error_message = reset_dataframe_index(df)
                if new_error_message is None:
                    session['code_log'].append("df = df.reset_index(drop=True)")
                    success_message = "DataFrame index has been reset."

            if new_error_message is None:

                _save_df_to_project(df)
                _save_project_state()

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

    df, error_message = _get_df_from_project()

    if error_message:
        return redirect(url_for('index'))

    df_head_html, info_str, desc_html, columns, unique_values = get_dataframe_summary(df)
    plot_json = None
    new_error_message = None

    if request.method == 'POST':
        try:
            analysis_type = request.form.get('analysis_type')

            color_col = request.form.get('color_col') or None # Use None if empty


            if analysis_type == 'univariate':
                column = request.form.get('uni_column')
                if column:

                    plot_json = generate_univariate_plot(df, column, color=color_col)

                else:
                    new_error_message = "Please select a column for univariate analysis."

            elif analysis_type == 'bivariate':
                x_col = request.form.get('bi_x_column')
                y_col = request.form.get('bi_y_column')
                if x_col and y_col:

                    plot_json = generate_bivariate_plot(df, x_col, y_col, color=color_col)

                else:
                    new_error_message = "Please select two columns for bivariate analysis."

            elif analysis_type == 'multivariate':
                multi_columns = request.form.getlist('multi_columns')
                if len(multi_columns) > 1:

                    plot_json = generate_multivariate_plot(df, multi_columns, color=color_col)

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


@app.route('/data_filtering', methods=['GET', 'POST'])
def data_filtering():

    df, error_message = _get_df_from_project()

    if error_message:
        return redirect(url_for('index'))

    new_error_message = None
    success_message = None

    if request.method == 'POST':
        try:
            column = request.form.get('filter_col')
            operator = request.form.get('filter_op')
            value = request.form.get('filter_val')

            original_rows = len(df)
            df, new_error_message = filter_dataframe(df, column, operator, value)
            if new_error_message is None:

                _save_df_to_project(df)
                filtered_rows = len(df)
                success_message = f"Filter applied successfully. Showing {filtered_rows} of {original_rows} rows."
                session['code_log'].append(f"df = df[df['{column}'] {operator} '{value}'] # Adjust value quoting for strings vs. numbers")
                _save_project_state()

        except Exception as e:
            new_error_message = f"An error occurred during filtering: {e}"

    df_head_html = df.head(100).to_html(classes=['table', 'table-striped', 'table-sm'])
    _, _, columns, _ = get_dataframe_summary(df)

    # This page doesn't neatly fit in the linear pipeline, so we can handle progress differently
    # For now, let's just pass dummy values or handle it gracefully in the template
    current_stage = "Data Filtering"
    progress_percent = 50

    return render_template('data_filtering.html',
                           df_head=df_head_html, columns=columns,
                           error=new_error_message, success=success_message,
                           current_stage=current_stage, progress_percent=progress_percent)


@app.route('/data_engineering', methods=['GET', 'POST'])
def data_engineering():

    df, error_message = _get_df_from_project()

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
                    op_map = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}
                    session['code_log'].append(f"df['{new_col_name}'] = df['{col1}'] {op_map[operation]} df['{col2}']")
                    success_message = f"New feature '{new_col_name}' created."

            elif action == 'apply_encoding':
                columns = request.form.getlist('columns')
                encoding_type = request.form.get('encoding_type')
                df, new_error_message = apply_encoding(df, columns, encoding_type)
                if new_error_message is None:
                    session['code_log'].append(f"# Applying {encoding_type} encoding to {columns} (code is complex)")
                    success_message = f"Columns encoded using '{encoding_type}'."

            elif action == 'scale_features':
                columns = request.form.getlist('columns')
                scaler_type = request.form.get('scaler_type')
                df, new_error_message = scale_features(df, columns, scaler_type)
                if new_error_message is None:
                    session['code_log'].append(f"# Scaling {columns} using {scaler_type} scaler (code is complex)")
                    success_message = f"Columns scaled using '{scaler_type}'."

            elif action == 'rename_drop':
                old_col_name = request.form.get('old_column_name')
                new_col_name = request.form.get('new_column_name')
                columns_to_drop = request.form.getlist('columns_to_drop')
                df, new_error_message = rename_and_drop_columns(df, new_column_name, old_col_name, columns_to_drop)
                if new_error_message is None:
                    success_message = "Columns renamed/dropped."

            if new_error_message is None:

                _save_df_to_project(df)
                _save_project_state()

        except Exception as e:
            new_error_message = f"An error occurred: {e}"


    df_head_html, _, _, columns, _ = get_dataframe_summary(df)
    current_stage, progress_percent = _get_progress_data("Feature Engineering")
    return render_template('data_engineering.html',
                           df_head=df_head_html, columns=columns, error=new_error_message,
                           success=success_message, current_stage=current_stage,
                           progress_percent=progress_percent)


@app.route('/data_aggregation', methods=['GET', 'POST'])
def data_aggregation():

    df, error_message = _get_df_from_project()

    if error_message:
        return redirect(url_for('index'))

    new_error_message = None
    success_message = None
    result_df_head = None

    if request.method == 'POST':
        try:
            agg_type = request.form.get('agg_type')
            result_df = None

            if agg_type == 'groupby':
                groupby_cols = request.form.getlist('groupby_cols')
                agg_col = request.form.get('agg_col')
                agg_func = request.form.get('agg_func')
                result_df, new_error_message = group_by_aggregate(df, groupby_cols, agg_col, agg_func)
                if new_error_message is None:
                    success_message = "GroupBy aggregation successful."
                    session['code_log'].append(f"agg_df = df.groupby({groupby_cols})['{agg_col}'].agg('{agg_func}').reset_index()")

            elif agg_type == 'pivot':
                index_cols = request.form.getlist('pivot_index')
                column_cols = request.form.getlist('pivot_cols')
                value_col = request.form.get('pivot_val')
                agg_func = request.form.get('pivot_agg')
                result_df, new_error_message = pivot_table(df, index_cols, column_cols, value_col, agg_func)
                if new_error_message is None:
                    success_message = "Pivot table created successfully."
                    session['code_log'].append(f"pivot_df = df.pivot_table(values='{value_col}', index={index_cols}, columns={column_cols}, aggfunc='{agg_func}')")

            if result_df is not None:
                # Don't save the aggregated df, just show it
                result_df_head = result_df.head(100).to_html(classes=['table', 'table-striped', 'table-sm'])

        except Exception as e:
            new_error_message = f"An error occurred: {e}"

    _, _, columns, _ = get_dataframe_summary(df)
    current_stage = "Data Aggregation"
    progress_percent = 60

    return render_template('data_aggregation.html',
                           columns=columns, df=df, result_df_head=result_df_head,
                           error=new_error_message, success=success_message,
                           current_stage=current_stage, progress_percent=progress_percent)



@app.route('/data_combining', methods=['GET', 'POST'])
def data_combining():
    left_df, error_message = _get_df_from_project()

    if error_message:
        return redirect(url_for('index'))

    new_error_message = None
    success_message = None
    right_df_head = None
    result_df_head = None
    left_cols = left_df.columns.tolist()
    right_cols = []

    # Handle POST requests
    if request.method == 'POST':
        action = request.form.get('action')
        try:
            if action == 'upload_right_df':
                file = request.files.get('file_right')
                if file and file.filename:
                    right_df, new_error_message = load_data(source_type='upload', source_path_or_file=file)
                    if new_error_message is None:
                        # Save the right df to a temporary path in session
                        right_filename = f"right_{uuid.uuid4()}.csv"
                        right_filepath = os.path.join(app.config['UPLOAD_FOLDER'], right_filename)
                        right_df.to_csv(right_filepath, index=False)
                        session['filepath_right'] = right_filepath
                        success_message = "Second DataFrame uploaded successfully."
                else:
                    new_error_message = "No file selected for upload."

            elif action == 'perform_combine':
                if 'filepath_right' in session:
                    right_df = pd.read_csv(session['filepath_right'])
                    method = request.form.get('method')

                    params = {}
                    if method == 'merge':
                        params['left_on'] = request.form.getlist('left_on')
                        params['right_on'] = request.form.getlist('right_on')
                        params['how'] = request.form.get('how')
                    elif method == 'concat':
                        params['axis'] = int(request.form.get('axis', 0))

                    combined_df, new_error_message = combine_dataframes(left_df, right_df, method, **params)

                    if new_error_message is None:
                        # Overwrite the main df with the result

                        _save_df_to_project(combined_df)

                        result_df_head = combined_df.head().to_html(classes=['table', 'table-striped', 'table-sm'])
                        success_message = f"DataFrames combined successfully using '{method}'."
                        # Clean up the right dataframe from session and disk
                        os.remove(session['filepath_right'])
                        session.pop('filepath_right', None)
                else:
                    new_error_message = "Please upload the second DataFrame first."

        except Exception as e:
            new_error_message = f"An error occurred: {e}"

    # Prepare variables for GET request or re-rendering
    left_df_head = left_df.head().to_html(classes=['table', 'table-striped', 'table-sm'])
    if 'filepath_right' in session and os.path.exists(session['filepath_right']):
        right_df = pd.read_csv(session['filepath_right'])
        right_df_head = right_df.head().to_html(classes=['table', 'table-striped', 'table-sm'])
        right_cols = right_df.columns.tolist()

    return render_template('data_combining.html',
                           left_df_head=left_df_head, right_df_head=right_df_head,
                           left_cols=left_cols, right_cols=right_cols,
                           error=new_error_message, success=success_message,
                           result_df_head=result_df_head,
                           current_stage="Data Combining", progress_percent=30)



@app.route('/model_building', methods=['GET', 'POST'])
def model_building():
    df, error_message = _get_df_from_project()

    if error_message:
        return redirect(url_for('index'))

    if request.method == 'POST':
        try:
            target_col = request.form.get('target_col')
            feature_cols = request.form.getlist('feature_cols')
            problem_type = request.form.get('problem_type')
            test_size = float(request.form.get('test_size'))

            # The model checkboxes are disabled in HTML based on problem type,
            # so we only get the relevant ones.
            selected_models = request.form.getlist('models')

            if not target_col or not feature_cols or not selected_models:
                raise ValueError("Missing required fields: target, features, or models.")

            # Run the models
            results_df = run_models(df, feature_cols, target_col, problem_type, test_size)

            # Prepare results for rendering
            results_table = results_df.drop(columns=['Confusion Matrix'], errors='ignore').to_html(classes=['table', 'table-striped', 'table-sm'])
            results_list = results_df.to_dict(orient='records')


            # Save the run configuration for the tuning step
            session['last_run_features'] = feature_cols
            session['last_run_target'] = target_col
            session['last_run_problem_type'] = problem_type


            return render_template('model_results.html',
                                   results_table=results_table,
                                   results_list=results_list,
                                   problem_type=problem_type,
                                   current_stage="Model Evaluation",
                                   progress_percent=90)
        except Exception as e:
            # On error, re-render the model building page with the error message
            _, _, columns, _ = get_dataframe_summary(df)
            return render_template('model_building.html',
                                   columns=columns,
                                   models=get_model_list(),
                                   error=f"An error occurred: {e}",
                                   current_stage="Model Building",
                                   progress_percent=75)

    # For GET request
    _, _, columns, _ = get_dataframe_summary(df)
    current_stage, progress_percent = _get_progress_data("Model Building")
    return render_template('model_building.html',
                           columns=columns,
                           models=get_model_list(),
                           current_stage=current_stage,
                           progress_percent=progress_percent)



@app.route('/model_tuning/<model_name>', methods=['GET', 'POST'])
def model_tuning(model_name):
    df, error_message = _get_df_from_project()

    if error_message:
        return redirect(url_for('index'))

    # This is a simplified way to keep state. In a larger app, you'd pass this via forms.
    # For now, we assume the last run's settings are what we're tuning.
    # This part needs the session to be populated from the model_building run,
    # which we are not doing yet. This will be faked for now.
    # TODO: Pass features and target from model_building to this route.
    if 'last_run_features' not in session:
        # Fallback for now
        session['last_run_features'] = df.columns.tolist()[:-1]
        session['last_run_target'] = df.columns.tolist()[-1]
        session['last_run_problem_type'] = 'Classification' if df[session['last_run_target']].dtype == 'object' else 'Regression'


    features = session['last_run_features']
    target = session['last_run_target']
    problem_type = session['last_run_problem_type']

    param_grid_options = get_hyperparameter_grid()[problem_type].get(model_name, {})
    tuning_results = None
    new_error_message = None

    if request.method == 'POST':
        try:
            # Construct param_grid from form
            param_grid = {}
            for param, values in param_grid_options.items():
                user_values = request.form.get(param)
                if user_values:
                    # Super basic parsing: split by comma and try to convert to int/float
                    processed_values = []
                    for v in user_values.split(','):
                        v = v.strip()
                        try:
                            processed_values.append(float(v) if '.' in v else int(v))
                        except ValueError:
                            processed_values.append(v) # Keep as string if conversion fails
                    param_grid[param] = processed_values

            tuning_results = tune_model_hyperparameters(df, features, target, problem_type, model_name, param_grid)
        except Exception as e:
            new_error_message = f"An error occurred during tuning: {e}"


    return render_template('model_tuning.html',
                           model_name=model_name,
                           param_grid_options=param_grid_options,
                           tuning_results=tuning_results,
                           error=new_error_message,
                           current_stage="Model Improvement",
                           progress_percent=95)



@app.route('/user_guide')
def user_guide():
    """
    Renders the user guide page.
    """
    return render_template('user_guide.html')


@app.route('/export', methods=['GET', 'POST'])
def export():

    df, error_message = _get_df_from_project()

    if error_message:
        return redirect(url_for('index'))

    if request.method == 'POST':
        export_format = request.form.get('export_format')

        if export_format in ['csv', 'xlsx']:
            file_content, mime_type = export_dataframe(df, export_format)
            return send_file(
                io.BytesIO(file_content),
                mimetype=mime_type,
                as_attachment=True,
                download_name=f'exported_data.{export_format}'
            )
        elif export_format == 'ipynb':
            include_code = request.form.get('include_code') == 'true'
            code_log = session.get('code_log', []) if include_code else []
            df_head_html = df.head().to_html(classes=['table', 'table-striped', 'table-sm'])

            file_content, mime_type = export_ipynb(df_head_html, code_log, "Final")
            return send_file(
                io.BytesIO(file_content),
                mimetype=mime_type,
                as_attachment=True,
                download_name='analysis_notebook.ipynb'
            )

    df_head_html, _, _, _, _ = get_dataframe_summary(df)
    current_stage, progress_percent = _get_progress_data("Export & Finalization")
    return render_template('export.html',
                           df_head=df_head_html,
                           current_stage=current_stage,
                           progress_percent=progress_percent)


if __name__ == '__main__':
    app.run(debug=True)
