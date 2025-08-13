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
# Import the new function for data combining
from utils.data_combining import combine_dataframes

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure, random key in production
app.config['UPLOAD_FOLDER'] = 'uploads/'
# New configuration for storing DataFrames on the server
app.config['DATA_FOLDER'] = 'data/'

# Ensure the upload and data folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

def load_df_from_session(key='current_df'):
    """Helper function to load a DataFrame from a file path stored in the session."""
    if key in session and session[key]:
        filepath = session[key]
        try:
            # Use a more efficient format like Parquet
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            flash(f"Error loading DataFrame from file: {e}", "danger")
            return None
    return None

def save_df_to_session(df, key='current_df'):
    """Helper function to save a DataFrame to a file and store the path in the session."""
    if df is not None:
        # Generate a unique filename and save the DataFrame to disk
        filename = f"{uuid.uuid4()}.parquet"
        filepath = os.path.join(app.config['DATA_FOLDER'], filename)
        df.to_parquet(filepath)
        
        # Clean up the old file if it exists
        if key in session and session[key] and os.path.exists(session[key]):
            os.remove(session[key])
            
        # Store only the file path in the session
        session[key] = filepath

def generate_df_viewer(df, num_rows=5):
    """Helper function to generate an HTML table for viewing a DataFrame."""
    if df is not None:
        return df.head(num_rows).to_html(classes=['table', 'table-striped', 'table-hover', 'table-responsive'], border=0)
    return None

def save_plot_to_session(plot_data, plot_name):
    """Helper function to save a Plotly figure to the session as JSON."""
    session[plot_name] = json.dumps(plot_data, cls=go.utils.PlotlyJSONEncoder)

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
    projects = [] # Replace with actual project loading logic
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
        return redirect(url_for('data_cleaning'))

    columns = df.columns.tolist()
    data_viewer = generate_df_viewer(df)
    return render_template('data_cleaning.html', columns=columns, data_viewer=data_viewer)

@app.route('/data_eda', methods=['GET', 'POST'])
def data_eda():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))
    
    plot_json = session.get('plot_json', None)
    
    if request.method == 'POST':
        action = request.form.get('action')
        error = None
        plot_fig = None

        if action == 'univariate_plot':
            column = request.form.get('column')
            plot_type = request.form.get('plot_type')
            color_col = request.form.get('color') or None
            plot_fig, error = generate_univariate_plot(df, column, plot_type, color=color_col)
        elif action == 'bivariate_plot':
            x_col = request.form.get('x_col')
            y_col = request.form.get('y_col')
            plot_type = request.form.get('plot_type')
            color_col = request.form.get('color') or None
            plot_fig, error = generate_bivariate_plot(df, x_col, y_col, plot_type, color=color_col)

        if error:
            flash(error, 'danger')
            return redirect(url_for('data_eda'))
        
        if plot_fig:
            plot_json = plot_fig.to_json()
            session['plot_json'] = plot_json
        
        return redirect(url_for('data_eda'))

    columns = df.columns.tolist()
    return render_template('data_eda.html', columns=columns, plot_json=plot_json)

@app.route('/data_engineering', methods=['GET', 'POST'])
def data_engineering():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))

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
            columns = request.form.getlist('columns[]')
            encoding_type = request.form.get('encoding_type')
            new_df, error = apply_encoding(new_df, columns, encoding_type)
        elif action == 'bin_column':
            column = request.form.get('column')
            num_bins = request.form.get('num_bins')
            new_col_name = request.form.get('new_col_name')
            labels = request.form.get('labels')
            new_df, error = bin_column(new_df, column, int(num_bins), new_col_name, labels.split(',') if labels else None)
        
        if error:
            flash(error, 'danger')
        else:
            save_df_to_session(new_df)
            flash(f'Data engineering action "{action}" applied successfully!', 'success')
        return redirect(url_for('data_engineering'))
    
    columns = df.columns.tolist()
    data_viewer = generate_df_viewer(df)
    return render_template('data_engineering.html', columns=columns, data_viewer=data_viewer)

@app.route('/data_aggregation', methods=['GET', 'POST'])
def data_aggregation():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))
    
    aggregated_df = None
    if request.method == 'POST':
        group_by_cols = request.form.getlist('group_by_cols[]')
        agg_col = request.form.get('agg_col')
        agg_func = request.form.getlist('agg_func[]')
        
        result_df, error = group_by_aggregate(df, group_by_cols, agg_col, agg_func)
        
        if error:
            flash(error, 'danger')
        else:
            aggregated_df = generate_df_viewer(result_df, num_rows=10)
            session['aggregated_df'] = result_df.to_json(orient='split') # Storing for viewing
            flash('Aggregation successful!', 'success')
    
    columns = df.columns.tolist()
    agg_functions = ['sum', 'mean', 'count', 'min', 'max'] # Example list of functions
    
    return render_template('data_aggregation.html', columns=columns, agg_functions=agg_functions, aggregated_df=aggregated_df)

@app.route('/data_filtering', methods=['GET', 'POST'])
def data_filtering():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))
    
    filtered_df_html = None
    if request.method == 'POST':
        column = request.form.get('column')
        operator = request.form.get('operator')
        value = request.form.get('value')
        
        new_df, error = filter_dataframe(df, column, operator, value)
        
        if error:
            flash(error, 'danger')
        else:
            save_df_to_session(new_df)
            filtered_df_html = generate_df_viewer(new_df, num_rows=10)
            flash('Filter applied successfully!', 'success')
        
    columns = df.columns.tolist()
    
    return render_template('data_filtering.html', columns=columns, filtered_df=filtered_df_html)

@app.route('/data_combining', methods=['GET', 'POST'])
def data_combining():
    # Load multiple dataframes using the new method
    df_paths = session.get('dataframes', {})
    
    # Placeholder for multiple dataframes
    if 'dataframes' not in session:
        session['dataframes'] = {}

    df_names = list(session['dataframes'].keys())
    combined_df_viewer = None
    error = None

    if request.method == 'POST':
        left_df_name = request.form.get('left_df')
        right_df_name = request.form.get('right_df')
        method = request.form.get('method')
        
        left_df_path = df_paths.get(left_df_name)
        right_df_path = df_paths.get(right_df_name)
        
        if not left_df_path or not right_df_path:
            error = "Both DataFrames must be selected."
        else:
            left_df = pd.read_parquet(left_df_path)
            right_df = pd.read_parquet(right_df_path)
            
            kwargs = {}
            if method == 'merge':
                kwargs['on'] = request.form.get('on')
                kwargs['how'] = request.form.get('how')
            elif method == 'concat':
                kwargs['axis'] = int(request.form.get('axis'))

            result_df, combine_error = combine_dataframes(left_df, right_df, method, **kwargs)

            if combine_error:
                error = combine_error
            else:
                new_df_name = f"{left_df_name}_{right_df_name}_combined"
                # Save the new DataFrame and store its path
                filename = f"{uuid.uuid4()}.parquet"
                filepath = os.path.join(app.config['DATA_FOLDER'], filename)
                result_df.to_parquet(filepath)
                
                session['dataframes'][new_df_name] = filepath
                save_df_to_session(result_df)
                combined_df_viewer = generate_df_viewer(result_df, num_rows=10)
                flash(f"DataFrames combined successfully! New DataFrame '{new_df_name}' created.", 'success')
                return redirect(url_for('data_combining'))

    # Update columns for the merge_on dropdown
    columns = []
    current_df = load_df_from_session()
    if current_df is not None:
        columns = current_df.columns.tolist()

    return render_template('data_combining.html', dataframes=df_names, combined_df=combined_df_viewer, error=error, columns=columns)

@app.route('/model_building', methods=['GET', 'POST'])
def model_building():
    df = load_df_from_session()
    if df is None:
        flash('Please ingest data first.', 'warning')
        return redirect(url_for('index'))
    
    results = session.get('model_results_html', None)
    feature_importance_plot = session.get('feature_importance_plot', None)
    confusion_matrix_plot = session.get('confusion_matrix_plot', None)
    
    if request.method == 'POST':
        action = request.form.get('action')
        error = None
        
        if action == 'run_models':
            problem_type = request.form.get('problem_type')
            target = request.form.get('target')
            features = request.form.getlist('features[]')
            test_size = float(request.form.get('test_size'))
            random_state = int(request.form.get('random_state'))
            
            model_results, fi_plot, cm_plot, error = run_models(df, problem_type, target, features, test_size, random_state)
            
            if error:
                flash(error, 'danger')
            else:
                session['model_results_html'] = model_results.to_html(classes=['table', 'table-striped', 'table-hover'], border=0)
                save_plot_to_session(fi_plot, 'feature_importance_plot')
                save_plot_to_session(cm_plot, 'confusion_matrix_plot')
                flash('Models run successfully!', 'success')
        
        return redirect(url_for('model_building'))
    
    columns = df.columns.tolist()
    available_models = ['Logistic Regression', 'Random Forest', 'SVM'] # Example
    return render_template('model_building.html', columns=columns, available_models=available_models,
                           results=results, feature_importance_plot=feature_importance_plot,
                           confusion_matrix_plot=confusion_matrix_plot)

@app.route('/export', methods=['GET', 'POST'])
def export():
    df = load_df_from_session()
    if df is None:
        flash('No data to export.', 'warning')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        file_format = request.form.get('file_format')
        
        # The file is saved and returned directly
        file_content, mime_type = export_dataframe(df, file_format)
        
        if file_content:
            output_filename = f'exported_data_{str(uuid.uuid4())[:8]}.{file_format}'
            return send_file(io.BytesIO(file_content), as_attachment=True, mimetype=mime_type, download_name=output_filename)
        else:
            flash(mime_type, 'danger')
            return redirect(url_for('export'))
        
    return render_template('export.html')

@app.route('/user_guide')
def user_guide():
    return render_template('user_guide.html')

if __name__ == '__main__':
    app.run(debug=True)
