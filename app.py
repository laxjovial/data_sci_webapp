#the main entry point 
# app.py

from flask import Flask, render_template, request, redirect, url_for, session
from utils.data_ingestion import load_data
import pandas as pd
import numpy as np
from utils.data_ingestion import load_data, get_dataframe_summary
from utils.data_cleaning import handle_missing_values, rename_column, convert_dtype

# app.py (add this after the imports)

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


app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here' # For session management

# A global variable to store the DataFrame across user sessions
# NOTE: In a production app, we would use a more robust solution, but this is fine for now.
# We will use the Flask session to handle this for a single user's experience.

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    The main route for the web application.
    
    Handles both GET and POST requests. The GET request renders the initial
    home page with options to upload data. The POST request processes the
    submitted data source and redirects to the data viewing page upon success.
    
    Technical: The route uses Flask's `request.method` to differentiate between
    GET and POST requests. For POST, it retrieves form data, calls the
    `load_data` utility, and stores the resulting DataFrame in the Flask session.
    This allows the data to persist between page redirects for a single user.
    
    Layman: This is the app's starting point. When you first visit, it shows
    you the page where you can upload your data. Once you submit a file or link,
    it grabs that data and sends you to the next page to see it.
    """
    if request.method == 'POST':
        source_type = request.form.get('source_type')
        source_path = request.form.get('source_path')

        if not source_path:
            return render_template('index.html', error="Please provide a valid file path or URL.")

        df, error_message = load_data(source_type, source_path)
        
        if error_message:
            return render_template('index.html', error=error_message)

        # Store DataFrame in session to access it in other routes
        session['df'] = df.to_json()
        return redirect(url_for('data_viewer'))

    return render_template('index.html')

@app.route('/data_viewer')
def data_viewer():
    """
    Displays the loaded dataset to the user.
    
    This page shows key information about the dataset and allows users to
    view different parts of the DataFrame (head, tail, unique values).
    
    Technical: It retrieves the DataFrame from the Flask session, converts
    it from JSON back to a pandas DataFrame, and then calculates various
    statistics like .info(), .describe(), and unique values for display.
    It uses a helper function `_get_df_from_session()` to ensure the DataFrame
    is always available.
    
    Layman: This is your "control panel" for the data. After you upload it,
    this page lets you see what's inside. You can check the first few rows,
    get a summary of the data, and see a list of all the columns.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        return render_template('index.html', error=error_message)

    # Convert pandas info to a readable string
    info_buffer = pd.io.common.StringIO()
    df.info(buf=info_buffer)
    info_str = info_buffer.getvalue()

    # Get descriptive statistics
    desc_html = df.describe(include='all').to_html(classes='table table-striped table-bordered')

    # Get unique values for each column
    unique_values = {col: df[col].unique().tolist()[:10] for col in df.columns}
    
    # Get the head of the DataFrame
    df_head_html = df.head().to_html(classes='table table-striped table-bordered')

    return render_template('data_viewer.html', 
                           df_head=df_head_html,
                           df_info=info_str,
                           df_desc=desc_html,
                           columns=df.columns.tolist(),
                           unique_values=unique_values)

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


@app.route('/data_cleaning')
def data_cleaning():
    """
    Renders the data cleaning page.

    Technical: Retrieves the current DataFrame from the session and prepares
    the data for display in the cleaning interface. It shows a preview of
    the data along with the available cleaning options.

    Layman: This is the data "car wash" page. It shows you your data and gives
    you tools to clean it, like fixing missing numbers or changing column names.
    """
    df, error_message = _get_df_from_session()
    if error_message:
        return render_template('index.html', error=error_message)

    df_head_html, _, _, columns, _ = get_dataframe_summary(df)
    return render_template('data_cleaning.html',
                           df_head=df_head_html,
                           columns=columns)


@app.route('/clean_data', methods=['POST'])
def clean_data():
    """
    Processes the data cleaning requests from the form.

    Technical: This route handles various cleaning actions based on the form
    submission. It calls the appropriate function from `utils/data_cleaning.py`,
    updates the DataFrame in the session, and redirects the user back to the
    cleaning page with the new, cleaned data.

    Layman: This is the engine that does the cleaning work. When you choose a
    cleaning option and click "Apply," this part of the app makes the changes
    and shows you the result right away.
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

    session['df'] = df.to_json()
    
    # Redirect back to the cleaning page
    df_head_html, _, _, columns, _ = get_dataframe_summary(df)
    return render_template('data_cleaning.html',
                           df_head=df_head_html,
                           columns=columns,
                           error=new_error_message)


if __name__ == '__main__':
    app.run(debug=True)


