import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

def generate_univariate_plot(df, column, plot_type='histogram', color=None):
    """Generates a univariate plot for a given column."""
    title = f'Univariate Analysis of {column}'
    if color:
        title += f' by {color}'
    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            if plot_type == 'histogram':
                fig = px.histogram(df, x=column, color=color, title=title, marginal='box', barmode='overlay')
            elif plot_type == 'violin':
                fig = px.violin(df, y=column, color=color, title=title, box=True, points="all")
            else:
                fig = px.histogram(df, x=column, color=color, title=title, marginal='box', barmode='overlay')
            fig.update_layout(bargap=0.1)
        else:
            if plot_type == 'pie':
                data = df[column].value_counts().reset_index()
                data.columns = [column, 'count']
                fig = px.pie(data, names=column, values='count', title=title)
            else:
                if color:
                    data = df.groupby([column, color]).size().reset_index(name='count')
                    fig = px.bar(data, x=column, y='count', color=color, title=title, barmode='group')
                else:
                    data = df[column].value_counts().reset_index()
                    data.columns = [column, 'count']
                    fig = px.bar(data, x=column, y='count', title=title)
        return fig
    except Exception as e:
        print(f"Error generating univariate plot: {e}")
        return None

def generate_bivariate_plot(df, x_col, y_col, plot_type='scatter', color=None):
    """Generates a bivariate plot for two columns."""
    title = f'Bivariate Analysis of {x_col} vs {y_col}'
    try:
        if plot_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, color=color, title=title, trendline='ols' if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]) else None)
        elif plot_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, color=color, title=title)
        elif plot_type == 'bar':
            fig = px.bar(df, x=x_col, y=y_col, color=color, title=title)
        else:
            fig = px.scatter(df, x=x_col, y=y_col, color=color, title=title)
        return fig
    except Exception as e:
        print(f"Error generating bivariate plot: {e}")
        return None

def generate_multivariate_plot(df, plot_type='correlation_heatmap'):
    """
    Generates a multivariate plot.

    Technical: Creates a correlation heatmap using Seaborn for numerical columns
    or a pair plot to visualize pairwise relationships in the dataset.

    Layman: This helps you see the relationships between many variables at once.
    The heatmap shows which variables are strongly related, while the pair plot
    gives you a grid of scatter plots for every combination of variables.

    Args:
        df (pd.DataFrame): The input DataFrame.
        plot_type (str): The type of plot ('correlation_heatmap' or 'pairplot').

    Returns:
        str: An HTML string containing the plot (as a base64 encoded image for seaborn plots)
             or a Plotly JSON object.
    """
    try:
        if plot_type == 'correlation_heatmap':
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.shape[1] < 2:
                return None, "Error: Correlation heatmap requires at least two numerical columns."

            plt.figure(figsize=(12, 10))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Heatmap', fontsize=16)

            # Save plot to a string buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return f'<img src="data:image/png;base64,{img_base64}" class="img-fluid"/>', None

        elif plot_type == 'pairplot':
            # Pairplot can be slow on large datasets, consider sampling or warning the user
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.shape[1] > 10: # Limit to 10 columns for performance
                numeric_df = numeric_df.iloc[:, :10]
            fig = px.scatter_matrix(numeric_df, title="Pair Plot of Numerical Features")
            return fig.to_json(), None
        else:
            return None, "Error: Invalid multivariate plot type."
    except Exception as e:
        return None, f"Error generating multivariate plot: {e}"

def generate_eda_report(df):
    """
    Generates a basic automated EDA report as an HTML string.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        str: An HTML string containing the EDA report.
    """
    # Basic Info
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()

    # Descriptive Statistics
    desc_html = df.describe().to_html(classes=['table', 'table-sm'])

    # Missing Values
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
