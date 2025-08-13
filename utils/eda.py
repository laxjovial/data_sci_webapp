import pandas as pd
import plotly.express as px
import plotly.io as pio


def generate_univariate_plot(df, column, plot_type='histogram', color=None):
    """
    Generates a univariate plot for a given column.
    
    Technical: This function uses the Plotly Express library to create
    interactive plots. It automatically detects if the data in a column is
    numeric or categorical and generates an appropriate visualization,
    such as a histogram, box plot, violin plot, or bar chart.

    Layman: This function helps you understand a single column of your data.
    For example, you can use it to see the distribution of ages in a dataset,
    or how many people fall into different income brackets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to visualize.
        plot_type (str): The type of plot to generate ('histogram', 'violin', 'bar', 'pie').
        color (str, optional): A column to use for color encoding.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    title = f'Univariate Analysis of {column}'

    if color:
        title += f' by {color}'

    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            if plot_type == 'histogram':
                fig = px.histogram(df, x=column, color=color, title=title, marginal='box', barmode='overlay')
            elif plot_type == 'violin':
                fig = px.violin(df, y=column, color=color, title=title, box=True, points="all")
            else:  # Default to histogram
                fig = px.histogram(df, x=column, color=color, title=title, marginal='box', barmode='overlay')
            fig.update_layout(bargap=0.1)
        else:  # Categorical
            if plot_type == 'pie':
                # Pie chart doesn't handle color grouping well, so we ignore it if selected
                data = df[column].value_counts().reset_index()
                data.columns = [column, 'count']
                fig = px.pie(data, names=column, values='count', title=title)
            else:  # Default to bar chart
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
    """
    Generates a bivariate plot for two columns.

    Technical: This function uses Plotly Express to create plots that show the
    relationship between two variables. It supports scatter plots for
    numeric-numeric relationships and bar charts for categorical-categorical
    or categorical-numeric relationships.

    Layman: This helps you understand the relationship between two different
    parts of your data. For example, you can see if there is a connection
    between a person's age and their salary.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The column for the x-axis.
        y_col (str): The column for the y-axis.
        plot_type (str): The type of plot to generate ('scatter', 'line', 'bar', etc.).
        color (str, optional): A column to use for color encoding.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
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
