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
                fig = px.bar(df[column].value_counts().reset_index(), x=df[column].name, y='count', title=title)
            fig.update_layout(xaxis_title=column, yaxis_title='Count')

    return pio.to_json(fig)

def generate_bivariate_plot(df, x_col, y_col, plot_type='scatter', color=None):
    """
    Generates a bivariate plot for two given columns.
    """
    title = f'Bivariate Analysis: {x_col} vs. {y_col}'
    if color:
        title += f' by {color}'
    
    try:
        # Both Numeric
        if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            if plot_type == 'scatter':
                fig = px.scatter(df, x=x_col, y=y_col, color=color, title=title, trendline='ols')
            elif plot_type == 'line':
                fig = px.line(df, x=x_col, y=y_col, color=color, title=title)
            else: # Default to scatter
                fig = px.scatter(df, x=x_col, y=y_col, color=color, title=title, trendline='ols')
        
        # Numeric vs. Categorical
        elif (pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_categorical_dtype(df[y_col])) or \
             (pd.api.types.is_categorical_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])):
            if plot_type == 'box':
                fig = px.box(df, x=x_col, y=y_col, color=color, title=title)
            elif plot_type == 'violin':
                fig = px.violin(df, x=x_col, y=y_col, color=color, title=title)
            else: # Default to box
                fig = px.box(df, x=x_col, y=y_col, color=color, title=title)
        
        # Both Categorical
        elif pd.api.types.is_categorical_dtype(df[x_col]) and pd.api.types.is_categorical_dtype(df[y_col]):
            # Cross-tabulation for count
            data = pd.crosstab(df[x_col], df[y_col])
            fig = px.imshow(data, x=data.columns, y=data.index, title=title, text_auto=True, color_continuous_scale='Viridis')
            fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        else:
            return None, "Error: Incompatible column types for bivariate plotting."

        return pio.to_json(fig), None

    except Exception as e:
        return None, f"An error occurred during bivariate plot generation: {e}"
