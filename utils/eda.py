import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def generate_univariate_plot(df, column, plot_type='histogram', color=None):
    """
    Generates a univariate plot from a Pandas DataFrame.

    Technical: This function samples a portion of the DataFrame (if it's large)
    to ensure responsiveness. It then creates plots with Plotly Express, which are
    optimized for in-memory data. The function handles both numeric and
    categorical data types with different plot types.

    Layman: This function helps you look at a single column of data. For example, you
    can see the distribution of ages with a histogram or the breakdown of genders with
    a bar or pie chart.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        column (str): The column to plot.
        plot_type (str): The type of plot to generate ('histogram', 'violin', 'pie', 'bar').
        color (str, optional): A column to use for coloring the plot.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
        None: If an error occurs.
    """
    if column not in df.columns:
        print(f"Error: Column '{column}' not found.")
        return None
        
    try:
        # Sample the data to make plotting feasible on large datasets
        sample_df = df.sample(frac=0.1) if len(df) > 50000 else df
        title = f'Univariate Analysis of {column}'
        if color:
            title += f' by {color}'

        if pd.api.types.is_numeric_dtype(sample_df[column]):
            if plot_type == 'histogram':
                fig = px.histogram(sample_df, x=column, color=color, title=title, marginal='box', barmode='overlay')
            elif plot_type == 'violin':
                fig = px.violin(sample_df, y=column, color=color, title=title, box=True, points="all")
            else:
                fig = px.histogram(sample_df, x=column, color=color, title=title, marginal='box', barmode='overlay')
            fig.update_layout(bargap=0.1)
        else: # Categorical data
            if plot_type == 'pie':
                data = sample_df[column].value_counts().reset_index()
                data.columns = [column, 'count']
                fig = px.pie(data, names=column, values='count', title=title)
            else:
                if color and color in sample_df.columns:
                    data = sample_df.groupby([column, color]).size().reset_index(name='count')
                    fig = px.bar(data, x=column, y='count', color=color, title=title, barmode='group')
                else:
                    data = sample_df[column].value_counts().reset_index()
                    data.columns = [column, 'count']
                    fig = px.bar(data, x=column, y='count', title=title)
        return fig
    except Exception as e:
        print(f"Error generating univariate plot: {e}")
        return None

def generate_bivariate_plot(df, x_col, y_col, plot_type='scatter', color=None):
    """
    Generates a bivariate plot from a Pandas DataFrame.

    Technical: Similar to the univariate function, this samples the DataFrame for
    plotting if it's large. It supports various plot types for visualizing the
    relationship between two columns.

    Layman: This helps you see how two columns of data relate to each other. For instance,
    you can use a scatter plot to see if there's a relationship between 'Age' and
    'Income', or a line plot to see how 'Temperature' changes over 'Time'.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        x_col (str): The column for the x-axis.
        y_col (str): The column for the y-axis.
        plot_type (str): The type of plot to generate ('scatter', 'line', 'bar').
        color (str, optional): A column to use for coloring the plot.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
        None: If an error occurs.
    """
    if x_col not in df.columns or y_col not in df.columns:
        print("Error: One or more specified columns not found.")
        return None
        
    try:
        cols_to_sample = [x_col, y_col]
        if color and color in df.columns:
            cols_to_sample.append(color)

        sample_df = df[cols_to_sample].sample(frac=0.1) if len(df) > 50000 else df
        title = f'Bivariate Analysis of {x_col} vs {y_col}'
        
        if plot_type == 'scatter':
            fig = px.scatter(sample_df, x=x_col, y=y_col, color=color, title=title,
                             trendline='ols' if pd.api.types.is_numeric_dtype(sample_df[x_col]) and pd.api.types.is_numeric_dtype(sample_df[y_col]) else None)
        elif plot_type == 'line':
            fig = px.line(sample_df, x=x_col, y=y_col, color=color, title=title)
        elif plot_type == 'bar':
            fig = px.bar(sample_df, x=x_col, y=y_col, color=color, title=title)
        else:
            fig = px.scatter(sample_df, x=x_col, y=y_col, color=color, title=title)
        return fig
    except Exception as e:
        print(f"Error generating bivariate plot: {e}")
        return None

def generate_multivariate_plot(df, plot_type='correlation_heatmap'):
    """
    Generates a multivariate plot from a Pandas DataFrame.

    Technical: This function generates a correlation heatmap for numerical columns
    using Seaborn, which is then returned as a base64 encoded image string.

    Layman: This helps you see the relationships between many variables at once. The
    heatmap shows which variables are strongly related.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        plot_type (str): The type of plot ('correlation_heatmap').

    Returns:
        tuple: A tuple containing an HTML string with the plot (as a base64 encoded image)
               and an error message (if any).
    """
    try:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
             return None, "Error: No numerical columns found for multivariate analysis."

        corr_matrix = df[numeric_cols].corr()

        if plot_type == 'correlation_heatmap':
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Heatmap', fontsize=16)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return f'<img src="data:image/png;base64,{img_base64}" class="img-fluid"/>', None

        else:
            return None, "Error: Invalid multivariate plot type."
    except Exception as e:
        return None, f"Error generating multivariate plot: {e}"

def generate_eda_report(df):
    """
    Generates a basic automated EDA report as an HTML string from a Pandas DataFrame.

    Technical: This function captures basic statistics (dtypes, describe, isnull) from
    the DataFrame and formats this information into a human-readable HTML string,
    providing a quick overview of the data.

    Layman: This gives you a quick summary of your data in a single view. It shows you
    things like what kind of data is in each column (text, numbers), key statistics
    like averages and ranges, and how many missing values are in each column.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.

    Returns:
        str: An HTML string containing the EDA report.
    """
    try:
        buf = io.StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()

        desc_html = df.describe().to_html(classes=['table', 'table-sm'])
        missing_values = df.isnull().sum().reset_index()
        missing_values.columns = ['Column', 'Missing Count']
        missing_html = missing_values[missing_values['Missing Count'] > 0].to_html(classes=['table', 'table-sm'], index=False)
        
        report_html = f"""
        <h4>DataFrame Info</h4>
        <pre>{info_str}</pre>
        <hr>
        <h4>Descriptive Statistics (Numerical)</h4>
        {desc_html}
        <hr>
        <h4>Missing Values</h4>
        {missing_html if not missing_values[missing_values['Missing Count'] > 0].empty else "<p>No missing values found.</p>"}
        """
        return report_html
    except Exception as e:
        print(f"Error generating EDA report: {e}")
        return f"An error occurred while generating the report: {e}"