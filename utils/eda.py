import pandas as pd
import plotly.express as px
import plotly.io as pio


def generate_univariate_plot(df, column, plot_type='histogram', color=None):
    """
    Generates a univariate plot for a given column.
    """
    title = f'Univariate Analysis of {column}'

    if color:
        title += f' by {color}'

    if pd.api.types.is_numeric_dtype(df[column]):

        if plot_type == 'histogram':
            fig = px.histogram(df, x=column, color=color, title=title, marginal='box', barmode='overlay')
        elif plot_type == 'violin':
            fig = px.violin(df, y=column, color=color, title=title, box=True, points="all")
        else: # Default to histogram
            fig = px.histogram(df, x=column, color=color, title=title, marginal='box', barmode='overlay')
        fig.update_layout(bargap=0.1)
    else: # Categorical
        if plot_type == 'pie':
             # Pie chart doesn't handle color grouping well, so we ignore it if selected
            data = df[column].value_counts().reset_index()
            data.columns = [column, 'count']
            fig = px.pie(data, names=column, values='count', title=title)
        else: # Default to bar chart
            if color:
                data = df.groupby([column, color]).size().reset_index(name='count')
                fig = px.bar(data, x=column, y='count', color=color, title=title, barmode='group')
            else:
                fig = px.bar(df[column].value_counts(), title=title)
            fig.update_layout(xaxis_title=column, yaxis_title='Count')

    return pio.to_json(fig)

def generate_bivariate_plot(df, x_col, y_col, plot_type='scatter', color=None):
    """
    Generates a bivariate plot for two given columns.
    """
    title = f'Bivariate Analysis: {x_col} vs. {y_col}'
    if color:
        title += f' by {color}'

    # Both Numeric
    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
        if plot_type == 'density_heatmap':
            fig = px.density_heatmap(df, x=x_col, y=y_col, title=title, marginal_x="histogram", marginal_y="histogram")
        else: # Default to scatter
            fig = px.scatter(df, x=x_col, y=y_col, color=color, title=title, trendline="ols")

    # One Numeric, One Categorical
    elif pd.api.types.is_numeric_dtype(df[x_col]) and not pd.api.types.is_numeric_dtype(df[y_col]):
        fig = px.box(df, x=y_col, y=x_col, color=color, title=title) # Box plot is standard here
    elif not pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
        fig = px.box(df, x=x_col, y=y_col, color=color, title=title) # Box plot is standard here

    # Both Categorical
    else:
        # Heatmap is standard here, color is ignored.

        crosstab = pd.crosstab(df[x_col], df[y_col])
        fig = px.imshow(crosstab, title=f'Heatmap of {x_col} vs. {y_col}', text_auto=True)

    return pio.to_json(fig)

def generate_multivariate_plot(df, columns, color=None):
    """
    Generates a multivariate plot for a list of columns.
    - Correlation heatmap for numeric columns (color is ignored).
    - Pair plot for a mix of columns (can be color-coded).
    """
    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]

    # Prioritize pair plot if a color dimension is chosen
    if color:
        title = f'Pair Plot of {", ".join(columns)} by {color}'
        fig = px.scatter_matrix(df[columns], color=color, title=title)
    # If no color, and enough numeric columns, show a heatmap
    elif len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, title='Correlation Matrix',
                        labels=dict(color="Correlation"))
    # Fallback to a simple pair plot
    else:
        fig = px.scatter_matrix(df[columns], title='Pair Plot')

    return pio.to_json(fig)
