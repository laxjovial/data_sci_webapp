#the main entry point 
# app.py

from flask import Flask, render_template, request, redirect, url_for, session
from utils.data_ingestion import load_data
import pandas as pd
import numpy as np

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

if __name__ == '__main__':
    app.run(debug=True)
