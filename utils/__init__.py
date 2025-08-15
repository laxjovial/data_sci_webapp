# utils/__init__.py

"""
This package contains all the core data processing and analysis functions
for the Data Science Web App.

By importing all the functions here, we can simplify the imports in the main
app.py file, making the code cleaner and more maintainable.
"""

# Import functions from each module to make them accessible at the package level.
from .data_ingestion import load_data
from .data_cleaning import (
    handle_missing_values,
    rename_column,
    convert_dtype,
    remove_duplicates,
    standardize_text,
    handle_outliers_iqr,
    apply_regex_cleaning,
    impute_knn
)
from .data_aggregation import group_by_aggregate, create_pivot_table
from .data_filtering import filter_dataframe
from .data_engineering import (
    create_new_feature,
    apply_encoding,
    bin_column,
    scale_features,
    create_polynomial_features
)
from .eda import (
    generate_univariate_plot,
    generate_bivariate_plot,
    generate_multivariate_plot,
    generate_eda_report
)
from .modeling import run_model, get_model_list, generate_shap_summary_plot
from .data_export import export_dataframe
from .data_combining import combine_dataframes
from .project_management import save_project, load_project, list_projects, delete_project

# You can also define an __all__ variable to specify what gets imported
# when a user does 'from utils import *'
__all__ = [
    'load_data',
    'handle_missing_values',
    'rename_column',
    'convert_dtype',
    'remove_duplicates',
    'standardize_text',
    'handle_outliers_iqr',
    'apply_regex_cleaning',
    'impute_knn',
    'group_by_aggregate',
    'create_pivot_table',
    'filter_dataframe',
    'create_new_feature',
    'apply_encoding',
    'bin_column',
    'scale_features',
    'create_polynomial_features',
    'generate_univariate_plot',
    'generate_bivariate_plot',
    'generate_multivariate_plot',
    'generate_eda_report',
    'run_model',
    'get_model_list',
    'generate_shap_summary_plot',
    'export_dataframe',
    'combine_dataframes',
    'save_project',
    'load_project',
    'list_projects',
    'delete_project'
]
