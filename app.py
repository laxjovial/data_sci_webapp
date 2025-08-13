# app.py

import os
import uuid
import json
import shutil
import io
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from werkzeug.utils import secure_filename
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# Import your corrected Python modules
from utils.data_ingestion import load_data
from utils.data_cleaning import handle_missing_values, rename_column, convert_dtype, remove_duplicates, standardize_text
from utils.data_aggregation import group_by_aggregate
from utils.data_filtering import filter_dataframe
from utils.data_engineering import create_new_feature, apply_encoding, bin_column
from utils.eda import generate_univariate_plot, generate_bivariate_plot
from utils.modeling import run_models
from utils.data_export import export_dataframe
from utils.data_combining import combine_dataframes

app = Flask(__name__)
# Change this to a secure, random key in production.
app.secret_key = 'your_secret_key'  
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DATA_FOLDER'] = 'data/'

# Ensure the upload and data folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

def load_df_from_session(key='current_df'):
    """Helper function to load a DataFrame from a file path stored in the session."""
    if key in session and session[key]:
        filepath = session[key]
        try:
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            flash(f"Error loading DataFrame from file: {e}", "danger")
            return None
    return None

def save_df_to_session(df, key='current_df'):
    """Helper function to save a DataFrame to a file and store the path in the session."""
    if df is not None:
        filename = f"{uuid.uuid4()}.parquet"
        filepath = os.path.join(app.config['DATA_FOLDER'], filename)
        df.to_parquet(filepath)
        
        # Clean up the old file if it exists
        if key in session and os.path.exists(session.get(key, '')):
            os.remove(session[key])
            
        session[key] = filepath

# CORRECTED FUNCTIONS TO HANDLE PLOT DATA
def load_plot_from_session(plot_name):
    """Helper function to load a Plotly figure from a file path stored in the session."""
    if plot_name in session and session[plot_name]:
        filepath = session[plot_name]
        try:
            with open(filepath, 'r') as f:
                plot_json = f.read()
            return plot_json
        except Exception as e:
            flash(f"Error loading plot from file: {e}", "danger")
            return None
    return None

def save_plot_to_session(fig, plot_name):
    """Helper function to save a Plotly figure to a file and store the path in the session."""
    if fig is not None:
        filename = f"{uuid.uuid4()}_{plot_name}.json"
        filepath = os.path.join(app.config['DATA_FOLDER'], filename)
        
        # Clean up old plot file if it exists
        if plot_name in session and os.path.exists(session.get(plot_name, '')):
            os.remove(session[plot_name])
            
        fig.write_json(filepath)
        session[plot_name] = filepath

def generate_df_viewer(df, num_rows=5):
    """Helper function to generate an HTML table for viewing a DataFrame."""
    if df is not None:
        return df.head(num_rows).to_html(classes=['table', 'table-striped', 'table-hover', 'table-responsive'], border=0)
    return None

@app.route('/')
def index():
    df = load_df_from_session()
    data_viewer = generate_df_viewer(df)
    return render_template('index.html', data_viewer=data_viewer)

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file_upload' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))
    file = request.files['file_upload']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        df, error = load_data(source_type='upload', source_path_or_file=filepath)
        if error:
            flash(error, 'danger')
            return redirect(url_for('index'))
        save_df_to_session(df)
        flash('File uploaded and data ingested successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/ingest_url', methods=['POST'])
def ingest_url():
    url = request.form.get('url')
    if not url:
        flash('No URL provided', 'danger')
        return redirect(url_for('index'))
    df, error = load_data(source_type='url', source_path_or_file=url)
    if error:
        flash(error, 'danger')
        return redirect(url_for('index'))
    save_df_to_session(df)
    flash('Data ingested from URL successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/projects')
def projects():
    # This route would list saved projects, which requires a database or file storage
    projects = [] 
    return render_template('projects.html', projects=projects)

@app.route('/data_cleaning', methods=['GET', 'POST'])
def data_cleaning():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))

    if request.method == 'POST':
        action = request.form.get('action')
        new_df = df.copy()
        error = None

        if action == 'handle_missing_values':
            columns = request.form.getlist('columns[]')
            strategy = request.form.get('strategy')
            new_df, error = handle_missing_values(new_df, columns, strategy)
        elif action == 'rename_column':
            old_col = request.form.get('old_col')
            new_col = request.form.get('new_col')
            new_df, error = rename_column(new_df, old_col, new_col)
        elif action == 'convert_dtype':
            column = request.form.get('column')
            new_type = request.form.get('new_type')
            new_df, error = convert_dtype(new_df, column, new_type)
        elif action == 'standardize_text':
            columns = request.form.getlist('columns[]')
            new_df, error = standardize_text(new_df, columns)
        elif action == 'remove_duplicates':
            new_df, error = remove_duplicates(new_df)

        if error:
            flash(error, 'danger')
        else:
            save_df_to_session(new_df)
            flash(f'Data cleaning action "{action}" applied successfully!', 'success')

    columns = df.columns.tolist()
    data_viewer = generate_df_viewer(df)
    
    # Check if a plot is available in the session
    plot_json = load_plot_from_session('current_plot')
    
    return render_template('data_cleaning.html', data_viewer=data_viewer, columns=columns, plot_json=plot_json)

@app.route('/eda_univariate', methods=['GET', 'POST'])
def eda_univariate():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))

    plot_json = None
    columns = df.columns.tolist()

    if request.method == 'POST':
        column = request.form.get('column')
        plot_type = request.form.get('plot_type', 'histogram')
        color = request.form.get('color')
        
        if column:
            fig = generate_univariate_plot(df, column, plot_type=plot_type, color=color)
            if fig:
                # Save the plot to a file and store the path in the session
                save_plot_to_session(fig, 'univariate_plot')
                plot_json = load_plot_from_session('univariate_plot')
            else:
                flash('Could not generate plot.', 'danger')
        else:
            flash('Please select a column.', 'warning')
    else:
        # Load any existing plot on GET request
        plot_json = load_plot_from_session('univariate_plot')

    data_viewer = generate_df_viewer(df)
    return render_template('eda_univariate.html', data_viewer=data_viewer, columns=columns, plot_json=plot_json)

@app.route('/eda_bivariate', methods=['GET', 'POST'])
def eda_bivariate():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))

    plot_json = None
    columns = df.columns.tolist()

    if request.method == 'POST':
        x_col = request.form.get('x_col')
        y_col = request.form.get('y_col')
        plot_type = request.form.get('plot_type', 'scatter')
        color = request.form.get('color')

        if x_col and y_col:
            fig = generate_bivariate_plot(df, x_col, y_col, plot_type, color=color)
            if fig:
                # Save the plot to a file and store the path in the session
                save_plot_to_session(fig, 'bivariate_plot')
                plot_json = load_plot_from_session('bivariate_plot')
            else:
                flash('Could not generate plot.', 'danger')
        else:
            flash('Please select two columns.', 'warning')
    else:
        # Load any existing plot on GET request
        plot_json = load_plot_from_session('bivariate_plot')
    
    data_viewer = generate_df_viewer(df)
    return render_template('eda_bivariate.html', data_viewer=data_viewer, columns=columns, plot_json=plot_json)

@app.route('/data_engineering', methods=['GET', 'POST'])
def data_engineering():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))

    columns = df.columns.tolist()
    
    if request.method == 'POST':
        action = request.form.get('action')
        new_df = df.copy()
        error = None

        if action == 'create_new_feature':
            col1 = request.form.get('col1')
            col2 = request.form.get('col2')
            operation = request.form.get('operation')
            new_col_name = request.form.get('new_col_name')
            new_df, error = create_new_feature(new_df, col1, col2, operation, new_col_name)
        elif action == 'apply_encoding':
            column = request.form.get('column')
            encoding_type = request.form.get('encoding_type')
            new_df, error = apply_encoding(new_df, column, encoding_type)
        elif action == 'bin_column':
            column = request.form.get('column')
            bins = request.form.get('bins')
            try:
                bins = int(bins)
                new_df, error = bin_column(new_df, column, bins)
            except (ValueError, TypeError):
                error = "Bins must be an integer."

        if error:
            flash(error, 'danger')
        else:
            save_df_to_session(new_df)
            flash(f'Data engineering action "{action}" applied successfully!', 'success')
            return redirect(url_for('data_engineering'))

    data_viewer = generate_df_viewer(df)
    return render_template('data_engineering.html', data_viewer=data_viewer, columns=columns)

@app.route('/data_aggregation', methods=['GET', 'POST'])
def data_aggregation():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))

    columns = df.columns.tolist()

    if request.method == 'POST':
        group_by_cols = request.form.getlist('group_by_cols[]')
        agg_col = request.form.get('agg_col')
        agg_func = request.form.get('agg_func')

        new_df, error = group_by_aggregate(df, group_by_cols, agg_col, agg_func)
        if error:
            flash(error, 'danger')
        else:
            save_df_to_session(new_df)
            flash('Data aggregated successfully!', 'success')
            return redirect(url_for('data_aggregation'))

    data_viewer = generate_df_viewer(df)
    return render_template('data_aggregation.html', data_viewer=data_viewer, columns=columns)

@app.route('/data_filtering', methods=['GET', 'POST'])
def data_filtering():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))

    columns = df.columns.tolist()

    if request.method == 'POST':
        column = request.form.get('column')
        operator = request.form.get('operator')
        value = request.form.get('value')
        
        new_df, error = filter_dataframe(df, column, operator, value)
        if error:
            flash(error, 'danger')
        else:
            save_df_to_session(new_df)
            flash('Data filtered successfully!', 'success')
            return redirect(url_for('data_filtering'))

    data_viewer = generate_df_viewer(df)
    return render_template('data_filtering.html', data_viewer=data_viewer, columns=columns)

@app.route('/data_combining', methods=['GET', 'POST'])
def data_combining():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))
    
    columns = df.columns.tolist()
    
    if request.method == 'POST':
        pass # Placeholder for future implementation
        # flash('This feature is not yet implemented.', 'info')
        # return redirect(url_for('data_combining'))
        
    data_viewer = generate_df_viewer(df)
    return render_template('data_combining.html', data_viewer=data_viewer, columns=columns)

@app.route('/modeling', methods=['GET', 'POST'])
def modeling():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))
    
    columns = df.columns.tolist()
    
    if request.method == 'POST':
        target_column = request.form.get('target_column')
        model_type = request.form.get('model_type')
        model_name = request.form.get('model_name')
        
        # Call the modeling function
        report = run_models(df, target_column, model_type, model_name)
        if report:
            session['modeling_report'] = report
            return redirect(url_for('modeling_results'))
        else:
            flash('Failed to run model.', 'danger')

    return render_template('modeling.html', columns=columns)

@app.route('/modeling_results')
def modeling_results():
    if 'modeling_report' in session:
        report = session.get('modeling_report')
        # Here we just pass the report directly.
        # In a real app, you might want to save and load it like the df and plots
        return render_template('modeling_results.html', report=report)
    else:
        flash('No modeling report found.', 'warning')
        return redirect(url_for('modeling'))

@app.route('/download/<file_format>', methods=['GET'])
def download(file_format):
    df = load_df_from_session()
    if df is None:
        flash('No data to download.', 'danger')
        return redirect(url_for('index'))

    data, error = export_dataframe(df, file_format)
    if error:
        flash(error, 'danger')
        return redirect(url_for('index'))
    
    if file_format == 'csv':
        mimetype = 'text/csv'
        filename = 'data.csv'
    elif file_format == 'excel':
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        filename = 'data.xlsx'
    else:
        flash('Invalid file format.', 'danger')
        return redirect(url_for('index'))
    
    return send_file(
        io.BytesIO(data),
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    app.run(debug=True)
