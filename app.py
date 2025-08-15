import os
import uuid
import json
import shutil
import io
import glob
import time
from threading import Timer

from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from werkzeug.utils import secure_filename
import pandas as pd
import dask.dataframe as dd
import plotly.graph_objs as go
import plotly.express as px

# Import all the necessary functions from the utils package
from utils import (
    load_data,
    handle_missing_values, rename_column, convert_dtype, remove_duplicates, standardize_text,
    handle_outliers_iqr, apply_regex_cleaning, impute_knn,
    group_by_aggregate, create_pivot_table,
    filter_dataframe,
    create_new_feature, apply_encoding, bin_column, scale_features, create_polynomial_features,
    generate_univariate_plot, generate_bivariate_plot, generate_multivariate_plot, generate_eda_report,
    run_model, get_model_list,
    export_dataframe,
    combine_dataframes,
    save_project, load_project, list_projects, delete_project
)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-for-development-only')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DATA_FOLDER'] = 'data/'
app.config['PROJECTS_DIR'] = 'projects/'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROJECTS_DIR'], exist_ok=True)

# --- Data File Cleanup ---
def cleanup_old_files(directory, max_age_hours=24):
    """Cleans up files in a directory older than a certain age."""
    now = time.time()
    max_age_seconds = max_age_hours * 3600

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.stat(filepath).st_mtime < now - max_age_seconds:
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
            print(f"Removed old file/dir: {filepath}")

def start_cleanup_scheduler():
    """Starts the periodic cleanup of old data and project files."""
    cleanup_old_files(app.config['DATA_FOLDER'])
    cleanup_old_files(app.config['PROJECTS_DIR'])
    # Rerun this function every 6 hours
    Timer(6 * 3600, start_cleanup_scheduler).start()

start_cleanup_scheduler()

# --- Dask-aware Session Management ---
def save_df_to_session(df, key='current_df_path'):
    """Saves a Dask DataFrame to Parquet and stores the path in the session."""
    if df is not None:
        dir_path = os.path.join(app.config['DATA_FOLDER'], f"{uuid.uuid4()}")
        if key in session and session.get(key) and os.path.exists(session[key]):
            shutil.rmtree(session[key])
        df.to_parquet(dir_path)
        session[key] = dir_path

def load_df_from_session(key='current_df_path'):
    """Loads a Dask DataFrame from a path stored in the session."""
    if key in session and session.get(key):
        dir_path = session[key]
        try:
            df = dd.read_parquet(dir_path)
            return df
        except Exception as e:
            flash(f"Error loading DataFrame from path: {e}", "danger")
            return None
    return None

def generate_df_viewer(df, num_rows=5):
    """Helper function to generate an HTML table for viewing a Dask DataFrame."""
    if df is not None:
        # Compute only the head for display
        return df.head(num_rows).to_html(classes=['table', 'table-striped', 'table-hover', 'table-responsive'], border=0)
    return None

def track_history(form_data):
    history = session.get('history', [])
    history.append(form_data)
    session['history'] = history

# --- Routes ---
@app.route('/')
def index():
    df = load_df_from_session()
    data_viewer = generate_df_viewer(df)
    return render_template('index.html', data_viewer=data_viewer)

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file_upload' not in request.files:
        flash("No file part in the request.", 'danger')
        return redirect(url_for('index'))
    
    file = request.files['file_upload']
    if file.filename == '':
        flash("No selected file.", 'danger')
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    try:
        df, error = load_data(filepath)
        if error:
            flash(error, 'danger')
            return redirect(url_for('index'))
        
        save_df_to_session(df)
        session['history'] = []
        flash('File uploaded and ingested as a scalable Dask DataFrame!', 'success')
    except Exception as e:
        flash(f"Error processing file: {e}", 'danger')

    return redirect(url_for('index'))


@app.route('/data_cleaning', methods=['GET', 'POST'])
def data_cleaning():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))

    if request.method == 'POST':
        action = request.form.get('action')
        new_df, error = df, None

        if action == 'handle_missing_values':
            new_df, error = handle_missing_values(df, request.form.getlist('columns[]'), request.form.get('strategy'))
        elif action == 'remove_duplicates':
            new_df, error = remove_duplicates(df, subset=request.form.getlist('subset[]'))
        # Add more cleaning actions here...

        if error:
            flash(error, 'danger')
        else:
            track_history(request.form.to_dict())
            save_df_to_session(new_df)
            flash(f'Action "{action}" applied.', 'success')
        return redirect(url_for('data_cleaning'))

    columns = df.columns
    df_for_view = df.head()
    data_viewer = generate_df_viewer(df_for_view)
    
    return render_template('data_cleaning.html', data_viewer=data_viewer, columns=columns)


@app.route('/data_filtering', methods=['GET', 'POST'])
def data_filtering():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))

    if request.method == 'POST':
        column = request.form.get('column')
        operator = request.form.get('operator')
        value = request.form.get('value')
        value2 = request.form.get('value2')

        try:
            new_df, error = filter_dataframe(df, column, operator, value, value2)
            if error:
                flash(error, 'danger')
                return redirect(url_for('data_filtering'))

            track_history(request.form.to_dict())
            save_df_to_session(new_df)
            flash('Data filtering applied successfully!', 'success')
        
        except Exception as e:
            flash(f'An error occurred during filtering: {e}', 'danger')
        
        return redirect(url_for('data_filtering'))

    columns = df.columns
    df_for_view = df.head()
    data_viewer = generate_df_viewer(df_for_view)
    
    return render_template('data_filtering.html', data_viewer=data_viewer, columns=columns)

@app.route('/data_aggregation', methods=['GET', 'POST'])
def data_aggregation():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))

    if request.method == 'POST':
        action = request.form.get('action')
        # Placeholder for Dask-aware aggregation logic
        flash(f"Action '{action}' is not yet refactored for Dask.", 'info')
        return redirect(url_for('data_aggregation'))

    columns = df.columns
    df_for_view = df.head()
    data_viewer = generate_df_viewer(df_for_view)
    
    return render_template('data_aggregation.html', data_viewer=data_viewer, columns=columns)

@app.route('/data_engineering', methods=['GET', 'POST'])
def data_engineering():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))

    if request.method == 'POST':
        action = request.form.get('action')
        new_df, error = None, None

        if action == 'scale_features':
            new_df, error = scale_features(df, request.form.getlist('scale_columns[]'), request.form.get('scaler_type'))

        if error:
            flash(error, 'danger')
        else:
            track_history(request.form.to_dict())
            save_df_to_session(new_df)
            flash(f'Action "{action}" applied.', 'success')
        return redirect(url_for('data_engineering'))

    columns = df.columns
    df_for_view = df.head()
    data_viewer = generate_df_viewer(df_for_view)
    
    return render_template('data_engineering.html', data_viewer=data_viewer, columns=columns)

@app.route('/data_eda', methods=['GET', 'POST'])
def data_eda():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Placeholder for EDA logic, which would need to be Dask-aware or use .compute() carefully
        flash("EDA generation is not yet refactored for Dask.", 'info')
        return redirect(url_for('data_eda'))

    columns = df.columns
    df_for_view = df.head()
    data_viewer = generate_df_viewer(df_for_view)
    
    return render_template('data_eda.html', data_viewer=data_viewer, columns=columns)


@app.route('/modeling', methods=['GET', 'POST'])
def modeling():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # For modeling, we need to compute the data to pass to scikit-learn
        pandas_df = df.compute()
        report = run_model(pandas_df, request.form.get('target_column'), request.form.get('model_type'), request.form.get('model_name'))
        
        if 'error' in report:
            flash(report['error'], 'danger')
        else:
            session['modeling_report'] = report
        return redirect(url_for('modeling'))

    report = session.get('modeling_report')
    return render_template('model_building.html', columns=df.columns, models=get_model_list(), report=report)

@app.route('/projects', methods=['GET', 'POST'])
def projects():
    if request.method == 'POST':
        action = request.form.get('action')
        project_name = request.form.get('project_name')

        if action == 'save':
            df = load_df_from_session()
            if df is not None:
                success, message = save_project(project_name, df, session.get('history', []))
                flash(message, 'success' if success else 'danger')

        elif action == 'load':
            df, history, error = load_project(project_name)
            if error:
                flash(error, 'danger')
            else:
                save_df_to_session(df)
                session['history'] = history
                flash(f"Project '{project_name}' loaded.", 'success')
                return redirect(url_for('index'))

        elif action == 'delete':
            success, message = delete_project(project_name)
            flash(message, 'success' if success else 'danger')
        
        return redirect(url_for('projects'))

    projects_list = list_projects()
    return render_template('projects.html', projects=projects_list)

@app.route('/download/<file_format>', methods=['GET'])
def download(file_format):
    df = load_df_from_session()
    if df is None:
        flash('No data to download.', 'warning')
        return redirect(url_for('index'))

    # For download, we need to compute the result and convert to pandas
    pandas_df = df.compute()
    data, error = export_dataframe(pandas_df, file_format)

    if error:
        flash(error, 'danger')
        return redirect(url_for('index'))
    
    mimetypes = {
        'csv': 'text/csv',
        'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'json': 'application/json',
        'parquet': 'application/octet-stream'
    }
    filenames = {
        'csv': 'data.csv',
        'excel': 'data.xlsx',
        'json': 'data.json',
        'parquet': 'data.parquet'
    }

    if file_format not in mimetypes:
        flash('Invalid file format.', 'danger')
        return redirect(url_for('index'))

    mimetype = mimetypes[file_format]
    filename = filenames[file_format]
    
    return send_file(
        io.BytesIO(data),
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    app.run(debug=True)