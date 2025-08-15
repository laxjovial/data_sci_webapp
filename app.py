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
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.stat(filepath).st_mtime < now - max_age_seconds:
            shutil.rmtree(filepath) if os.path.isdir(filepath) else os.remove(filepath)

def start_cleanup_scheduler():
    Timer(6 * 3600, start_cleanup_scheduler).start()
    cleanup_old_files(app.config['DATA_FOLDER'])
    cleanup_old_files(app.config['PROJECTS_DIR'])

start_cleanup_scheduler()

# --- Dask-aware Session Management ---
def save_df_to_session(df, key='current_df_path'):
    dir_path = os.path.join(app.config['DATA_FOLDER'], f"{uuid.uuid4()}")
    if key in session and session.get(key) and os.path.exists(session[key]):
        shutil.rmtree(session[key])
    df.to_parquet(dir_path)
    session[key] = dir_path

def load_df_from_session(key='current_df_path'):
    if key in session and session.get(key):
        return dd.read_parquet(session[key])
    return None

def generate_df_viewer(df, num_rows=5):
    if df is not None:
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
    if 'file_upload' not in request.files: return redirect(url_for('index'))
    file = request.files['file_upload']
    if not file or file.filename == '': return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    df, error = load_data(filepath) # load_data will now return a Dask df
    if error:
        flash(error, 'danger')
    else:
        save_df_to_session(df)
        session['history'] = []
        flash('File uploaded and ingested as a scalable Dask DataFrame!', 'success')
    return redirect(url_for('index'))

@app.route('/data_cleaning', methods=['GET', 'POST'])
def data_cleaning():
    df = load_df_from_session()
    if df is None: return redirect(url_for('index'))

    if request.method == 'POST':
        action = request.form.get('action')
        new_df, error = df, None

        if action == 'handle_missing_values':
            new_df, error = handle_missing_values(df, request.form.getlist('columns[]'), request.form.get('strategy'))
        # ... other actions ...

        if error:
            flash(error, 'danger')
        else:
            save_df_to_session(new_df)
            track_history(request.form.to_dict())
            flash(f'Action "{action}" applied.', 'success')
        return redirect(url_for('data_cleaning'))

    df_computed = df.head()
    return render_template('data_cleaning.html',
                           columns=df.columns,
                           df=df_computed,
                           data_viewer=generate_df_viewer(df_computed))

# ... All other data manipulation routes (data_filtering, data_combining, etc.)
# would follow the same pattern as data_cleaning above.
# They load the dask df, apply the dask-aware util function, save the new dask df,
# and for the GET request, compute the head for display.

@app.route('/data_engineering', methods=['GET', 'POST'])
def data_engineering():
    df = load_df_from_session()
    if df is None: return redirect(url_for('index'))

    if request.method == 'POST':
        action = request.form.get('action')
        new_df, error = df, None
        
        # Example for one action
        if action == 'scale_features':
            new_df, error = scale_features(df, request.form.getlist('scale_columns[]'), request.form.get('scaler_type'))
        # ... other actions ...

        if error: flash(error, 'danger')
        else:
            save_df_to_session(new_df)
            track_history(request.form.to_dict())
            flash(f'Action "{action}" applied.', 'success')
        return redirect(url_for('data_engineering'))

    df_computed = df.head()
    return render_template('data_engineering.html', columns=df.columns, df=df_computed, data_viewer=generate_df_viewer(df_computed))

@app.route('/modeling', methods=['GET', 'POST'])
def modeling():
    df = load_df_from_session()
    if df is None: return redirect(url_for('index'))

    if request.method == 'POST':
        target_column = request.form.get('target_column')
        model_type = request.form.get('model_type')
        model_name = request.form.get('model_name')
        
        # For modeling, we need to compute the data to pass to scikit-learn
        pandas_df = df.compute()
        report = run_model(pandas_df, target_column, model_type, model_name)

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
                # Project saving with Dask means saving the directory path
                # The save_project util needs to be dask-aware
                success, message = save_project(project_name, df, session.get('history', []))
                flash(message, 'success' if success else 'danger')

        elif action == 'load':
            # load_project returns a dask dataframe
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

    return render_template('projects.html', projects=list_projects())

@app.route('/download/<file_format>', methods=['GET'])
def download(file_format):
    df = load_df_from_session()
    if df is None: return redirect(url_for('index'))

    pandas_df = df.compute()
    data, error = export_dataframe(pandas_df, file_format)

    if error:
        flash(error, 'danger')
        return redirect(url_for('index'))
    
    mimetypes = {'csv': 'text/csv', 'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'json': 'application/json', 'parquet': 'application/octet-stream'}
    filenames = {'csv': 'data.csv', 'excel': 'data.xlsx', 'json': 'data.json', 'parquet': 'data.parquet'}

    if file_format not in mimetypes:
        flash('Invalid file format.', 'danger')
        return redirect(url_for('index'))

    return send_file(io.BytesIO(data), mimetype=mimetypes[file_format], as_attachment=True, download_name=filenames[file_format])

if __name__ == '__main__':
    app.run(debug=True)
