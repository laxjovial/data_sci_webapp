# DataFlow Pro - Owner's Guide (Developer & Maintainer)

This guide provides technical details for developers who want to run, maintain, or extend the DataFlow Pro application.

## 1. Architecture Overview

DataFlow Pro is a Flask-based web application designed for scalability and a professional user experience.

-   **Backend:** Python with Flask.
-   **Frontend:** Bootstrap 5 with custom CSS for a modern look and feel. JavaScript is used for dynamic UI updates.
-   **Data Backend:** The application uses **Dask** to handle potentially large datasets. Dataframes are loaded as Dask DataFrames and processed in a memory-efficient way. Intermediate dataframes are cached on the server as collections of Parquet files.
-   **Scalability:**
    -   **Dask Integration:** Allows for out-of-core (on-disk) computation for datasets larger than RAM.
    -   **Asynchronous Cleanup:** A background thread automatically cleans up old data and project files to manage server storage.
-   **Modularity:** The core data science logic is organized into a `utils` package, with each module dedicated to a specific part of the data science pipeline.

## 2. Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set the Secret Key (Optional but Recommended):**
    For production environments, set a secret key as an environment variable.
    ```bash
    export SECRET_KEY='your-very-strong-secret-key'
    ```

5.  **Run the application:**
    ```bash
    flask run
    # Or for production:
    # gunicorn --workers 4 --bind 0.0.0.0:8000 app:app
    ```

## 3. Code Structure

-   `app.py`: The main Flask application. Contains all routes and manages the session state. It has been refactored to use Dask for the data backend.
-   `utils/`: Contains all the core data processing logic. Each function is designed to be modular and reusable.
    -   `data_ingestion.py`: Uses Dask to load data from various sources.
    -   `project_management.py`: Handles the logic for saving and loading project states.
    -   ...and all other data science pipeline modules.
-   `templates/`: Contains all Jinja2 HTML templates.
    -   `base.html`: The master template with the main layout and a comprehensive custom stylesheet.
-   `projects/`: The directory where saved projects (data + history) are stored.
-   `data/`: The directory used for caching intermediate Dask dataframes.
-   `uploads/`: Temporary storage for user-uploaded files.

## 4. Extending the Application

This application was built with extensibility in mind.

-   **Adding a new `util` function:**
    1.  Add your new function to the appropriate module in the `utils/` directory (e.g., add a new cleaning function to `data_cleaning.py`). Ensure your function is designed to work with Dask DataFrames.
    2.  Export the function by adding its name to `utils/__init__.py`.
    3.  Import the function in `app.py`.
-   **Adding a new UI feature:**
    1.  Add a new route in `app.py` to handle the feature's logic.
    2.  Create a corresponding template in the `templates/` directory.
    3.  Add a link to the new feature in the `templates/base.html` navigation bar.
-   **Switching to a Database Backend:**
    As outlined in the `README.md`, the session management functions (`save_df_to_session`, `load_df_from_session`) in `app.py` can be modified to use a SQL database instead of the file system for even greater scalability and persistence.
