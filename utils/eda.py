import pandas as pd
import plotly.express as px
import plotly.io as pio

def generate_univariate_plot(df, column):
    """
    Generates a univariate plot for a given column.
    - Histogram for numeric data.
    - Bar chart (countplot) for categorical data.
    """
    if pd.api.types.is_numeric_dtype(df[column]):
        fig = px.histogram(df, x=column, title=f'Distribution of {column}', marginal='box')
        fig.update_layout(bargap=0.1)
    else:
        fig = px.bar(df[column].value_counts(), title=f'Count of {column}')
        fig.update_layout(xaxis_title=column, yaxis_title='Count')

    return pio.to_json(fig)

def generate_bivariate_plot(df, x_col, y_col):
    """
    Generates a bivariate plot for two given columns.
    - Scatter plot for two numeric columns.
    - Box plot for one numeric and one categorical column.
    - Heatmap for two categorical columns.
    """
    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
        fig = px.scatter(df, x=x_col, y=y_col, title=f'{x_col} vs. {y_col}', trendline="ols")
    elif pd.api.types.is_numeric_dtype(df[x_col]) and not pd.api.types.is_numeric_dtype(df[y_col]):
        fig = px.box(df, x=y_col, y=x_col, title=f'{x_col} by {y_col}')
    elif not pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
        fig = px.box(df, x=x_col, y=y_col, title=f'{y_col} by {x_col}')
    else:
        # For two categorical columns, create a crosstab and then a heatmap
        crosstab = pd.crosstab(df[x_col], df[y_col])
        fig = px.imshow(crosstab, title=f'Heatmap of {x_col} vs. {y_col}', text_auto=True)

    return pio.to_json(fig)

def generate_multivariate_plot(df, columns):
    """
    Generates a multivariate plot for a list of columns.
    - Correlation heatmap for numeric columns.
    - Pair plot for a mix of columns.
    """
    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]

    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, title='Correlation Matrix',
                        labels=dict(color="Correlation"))
    else:
        # Fallback to pair plot if not enough numeric columns for a meaningful heatmap
        fig = px.scatter_matrix(df[columns], title='Pair Plot')

    return pio.to_json(fig)
