import dask.dataframe as dd
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

def generate_univariate_plot(df, column, plot_type='histogram'):
    """Generates a univariate plot from a Dask DataFrame."""
    # For plotting, we usually need to compute the data.
    # For large datasets, it's best to work with a sample.
    sample_df = df.sample(frac=0.1).compute() # Taking a 10% sample
    # The rest of the function remains the same as it uses pandas
    # ...
    fig = px.histogram(sample_df, x=column, title=f'Univariate Analysis of {column}')
    return fig

def generate_bivariate_plot(df, x_col, y_col, plot_type='scatter'):
    """Generates a bivariate plot from a Dask DataFrame."""
    sample_df = df[[x_col, y_col]].sample(frac=0.1).compute()
    fig = px.scatter(sample_df, x=x_col, y=y_col, title=f'Bivariate Analysis of {x_col} vs {y_col}')
    return fig

def generate_multivariate_plot(df, plot_type='correlation_heatmap'):
    """Generates a multivariate plot from a Dask DataFrame."""
    numeric_cols = df.select_dtypes(include='number').columns
    sample_df = df[numeric_cols].sample(frac=0.1).compute()

    if plot_type == 'correlation_heatmap':
        plt.figure(figsize=(12, 10))
        sns.heatmap(sample_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" class="img-fluid"/>', None
    # ... other plot types
    return None, "Plot type not implemented for Dask."

def generate_eda_report(df):
    """Generates an EDA report from a Dask DataFrame."""
    # This will trigger computation.
    with pd.option_context('display.max_rows', 100):
        info_str = str(df.dtypes)
    desc_html = df.describe().compute().to_html(classes=['table', 'table-sm'])
    missing_html = df.isnull().sum().compute().to_frame('Missing Count').to_html(classes=['table', 'table-sm'])
    
    return f"<h4>Data Types</h4><pre>{info_str}</pre><hr><h4>Descriptive Statistics</h4>{desc_html}<hr><h4>Missing Values</h4>{missing_html}"
