# utils/data_visualization.py
import plotly.express as px
import pandas as pd

def generate_plot(df, plot_type, x_col, y_col=None):
    """
    Generates an interactive plot using Plotly Express.
    
    Technical: This function takes a DataFrame and plot parameters to generate
    a JSON-serializable Plotly figure. It supports common plot types like 
    histograms, box plots, and scatter plots, providing robust handling for 
    different chart requirements.
    
    Layman: This is the app's "artist." You tell it what you want to see—like 
    a bar chart or a scatter plot—and which columns to use, and it will draw 
    a professional, interactive picture of your data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        plot_type (str): The type of plot to generate ('histogram', 'boxplot', 'scatter').
        x_col (str): The column for the x-axis.
        y_col (str, optional): The column for the y-axis. Defaults to None.

    Returns:
        tuple: A JSON-formatted plot and an error message (if any).
    """
    try:
        if plot_type == 'histogram':
            fig = px.histogram(df, x=x_col)
        elif plot_type == 'boxplot':
            fig = px.box(df, y=x_col)
        elif plot_type == 'scatter' and y_col:
            fig = px.scatter(df, x=x_col, y=y_col)
        else:
            return None, "Invalid plot type or missing Y-column for scatter plot."

        # Update the layout for a cleaner look
        fig.update_layout(
            title_text=f'{plot_type.capitalize()} of {x_col}',
            xaxis_title=x_col,
            yaxis_title=y_col if y_col else 'Count'
        )

        return fig.to_json(), None
    except Exception as e:
        return None, f"An error occurred while generating the plot: {e}"
