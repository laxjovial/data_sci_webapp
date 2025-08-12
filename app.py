# app.py

from flask import Flask, render_template, request, redirect, url_for, session, send_file
from utils.data_ingestion import load_data, get_dataframe_summary
from utils.data_cleaning import (
    handle_missing_values, rename_column, convert_dtype, 
    remove_duplicates, standardize_text, handle_outliers, correct_inconsistencies
)
from utils.data_visualization import generate_plot
from utils.data_engineering import create_new_feature, one_hot_encode_column
from utils.data_export import export_dataframe, export_ipynb # New Import!
import pandas as pd
import numpy as np
import io

app = Flask(__name__)
# It's better to store the secret key in an environment variable for production
app.secret_key = 'your_super_secret_key_here' 

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
            # Use compression to handle larger dataframes within the session limit
            df_json_str = session['df']
            df = pd.read_json(df_json_str)
            return df, None
        except Exception as e:
            return None, f"Error restoring DataFrame from session: {e}"
    return None, "No dataset loaded. Please go back to the home page to upload one."

def _get_progress_data(current_stage_name):
    """
    Calculates the current progress based on the pipeline stages.
    
    Returns:
        tuple: (current_stage_name, progress_percentage)
    """
    try:
        current_index = PIPELINE_STAGES.index(current_stage_name)
        progress_percent = int(((current_index + 1) / len(PIPELINE_STAGES)) * 100)
        return current_stage_name, progress_percent
    except ValueError:
        return "Unknown Stage", 0

# --- NEW ROUTES FOR EXPORTING DATA ---
@app.route('/export_data', methods=['POST'])
def export_data():
    """
    Handles the request to export the current DataFrame.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        return redirect(url_for('index'))

    file_format = request.form.get('file_format')
    
    if file_format == 'ipynb':
        df_head_html, _, _, _, _ = get_dataframe_summary(df)
        current_stage, _ = _get_progress_data(request.form.get('current_stage'))
        
        # This is a placeholder for a real code log. 
        # You'll need to store and retrieve the actual code log from the session.
        code_log = [
            "# This is a placeholder code block.\n# You would store actual code from previous steps here.",
            "import pandas as pd\n\ndf = pd.read_csv('your_data.csv')"
        ]
        
        file_content, mime_type = export_ipynb(df_head_html, code_log, current_stage)
        if file_content:
            return send_file(io.BytesIO(file_content), 
                             mimetype=mime_type, 
                             as_attachment=True, 
                             download_name=f'analysis_{current_stage}.ipynb')
    else:
        file_content, mime_type = export_dataframe(df, file_format)
        if file_content:
            return send_file(io.BytesIO(file_content),
                             mimetype=mime_type,
                             as_attachment=True,
                             download_name=f'cleaned_data.{file_format}')
    
    return "Error during export.", 500


# --- EXISTING ROUTES (MODIFIED SLIGHTLY FOR EXPORT) ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    The main route for the web application, now with file upload functionality.
    """
    # ... (rest of the function is the same) ...
    if request.method == 'POST':
        source_type = request.form.get('source_type')
        
        if 'file' in request.files and request.files['file'].filename != '':
            source_file = request.files['file']
            df, error_message = load_data('upload', source_file)
        
        elif 'source_path' in request.form and request.form.get('source_path') != '':
            source_path = request.form.get('source_path')
            df, error_message = load_data('url', source_path)
            
        else:
            current_stage, progress_percent = _get_progress_data("Data Ingestion")
            return render_template('index.html', error="Please provide a valid file or URL.", current_stage=current_stage, progress_percent=progress_percent)

        if error_message:
            current_stage, progress_percent = _get_progress_data("Data Ingestion")
            return render_template('index.html', error=error_message, current_stage=current_stage, progress_percent=progress_percent)

        # We'll use a compressed JSON format to handle larger dataframes in session
        # This will help with the "file too large" issue you mentioned
        session['df'] = df.to_json(compression='infer')
        return redirect(url_for('data_viewer'))

    current_stage, progress_percent = _get_progress_data("Data Ingestion")
    return render_template('index.html', current_stage=current_stage, progress_percent=progress_percent)


@app.route('/data_viewer')
def data_viewer():
    # ... (rest of the function is the same) ...
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
    # ... (rest of the function is the same) ...
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
    # ... (rest of the function is the same) ...
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

    session['df'] = df.to_json(compression='infer')
    
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
    # ... (rest of the function is the same) ...
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
                           unique_values=unique_values, # Added this to support your EDA request
                           plot_json=plot_json,
                           error=error_message,
                           current_stage=current_stage,
                           progress_percent=progress_percent)


@app.route('/feature_engineering')
def feature_engineering():
    # ... (rest of the function is the same) ...
    df, error_message = _get_df_from_session()
    if error_message:
        return redirect(url_for('index'))

    df_head_html, _, _, columns, _ = get_dataframe_summary(df)
    
    current_stage, progress_percent = _get_progress_data("Feature Engineering")

    return render_template('feature_engineering.html',
                           df_head=df_head_html,
                           columns=columns,
                           current_stage=current_stage,
                           progress_percent=progress_percent)


@app.route('/engineer_features', methods=['POST'])
def engineer_features():
    # ... (rest of the function is the same) ...
    df, error_message = _get_df_from_session()
    if error_message:
        return redirect(url_for('index'))

    action_type = request.form.get('action_type')
    new_error_message = None

    if action_type == 'create_feature':
        col1 = request.form.get('col1')
        col2 = request.form.get('col2')
        new_col_name = request.form.get('new_col_name')
        operation = request.form.get('operation')
        if col1 and col2 and new_col_name and operation:
            df, new_error_message = create_new_feature(df, col1, col2, new_col_name, operation)
    
    elif action_type == 'one_hot_encode':
        column = request.form.get('encode_col')
        if column:
            df, new_error_message = one_hot_encode_column(df, column)

    session['df'] = df.to_json(compression='infer')
    
    df_head_html, _, _, columns, _ = get_dataframe_summary(df)
    current_stage, progress_percent = _get_progress_data("Feature Engineering")

    return render_template('feature_engineering.html',
                           df_head=df_head_html,
                           columns=columns,
                           error=new_error_message,
                           current_stage=current_stage,
                           progress_percent=progress_percent)

if __name__ == '__main__':
    app.run(debug=True)
