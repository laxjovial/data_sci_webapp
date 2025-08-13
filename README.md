# Comprehensive Data Science Web Application

This is a full-featured web application designed to streamline the entire data science workflow, from data ingestion and cleaning to exploratory data analysis (EDA), feature engineering, and eventually, model building. It is built with Flask and provides an interactive, user-friendly interface for users of all experience levels.

## Key Features

- **Flexible Data Ingestion:** Load data from your local device (CSV, Excel) or directly from web URLs, including automatic handling of GitHub and Google Drive links.
- **Robust Data Cleaning:** A suite of tools to handle common data quality issues:
    - Manage missing values with various strategies (mean, median, mode, drop).
    - Convert column data types.
    - Remove duplicate rows.
    - Standardize text (lowercase, uppercase, etc.).
    - Handle outliers using IQR methods.
    - Correct inconsistent values with a simple mapping dictionary.
- **Advanced Exploratory Data Analysis (EDA):**
    - **Univariate Analysis:** Explore individual variables through histograms and count plots.
    - **Bivariate Analysis:** Compare pairs of variables using scatter plots, box plots, and heatmaps.
    - **Multivariate Analysis:** Analyze relationships between multiple variables with correlation matrices and pair plots.
- **Powerful Feature Engineering:**
    - Create new features from existing ones using mathematical operations.
    - Encode categorical variables using various techniques (One-Hot, Label, Ordinal, Frequency).
    - Scale numerical features to a standard range (StandardScaler, MinMaxScaler).
- **Scalable Architecture:** The backend is built to handle large datasets by storing them on the server's filesystem, avoiding browser limitations.

## Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    The `requirements.txt` file contains all necessary Python packages.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

This is a Flask application. Once the dependencies are installed, you can run it using the following command:

```bash
flask run
```

The application will start, and you can access it by navigating to `http://127.0.0.1:5000` in your web browser.

## Application Structure

- `app.py`: The main Flask application file containing all routes and core logic.
- `requirements.txt`: A list of all Python dependencies.
- `uploads/`: A directory where uploaded data files are temporarily stored.
- `utils/`: A package containing the modular backend logic:
    - `data_ingestion.py`: Handles loading data from all sources.
    - `data_cleaning.py`: Contains functions for all data cleaning operations.
    - `data_engineering.py`: Contains functions for feature engineering tasks.
    - `eda.py`: Contains functions for generating all EDA plots.
- `templates/`: A directory containing all the HTML templates for the web interface.
    - `base.html`: The main layout template with the navigation and progress bar.
    - Other `.html` files correspond to the different pages/stages of the application.