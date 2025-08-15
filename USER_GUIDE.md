# DataFlow Pro - User Guide

Welcome to DataFlow Pro! This guide will walk you through how to use the application to perform a complete data science workflow, from loading data to building models.

## 1. Getting Started: Loading Your Data

The first step is always to load your data. You have two options on the main page:

-   **Upload a Dataset:** Click "Choose File" to select a file from your computer. The app supports CSV, Excel, JSON, and Parquet files.
-   **Ingest from URL:** Paste a direct link to a data file (e.g., a link to a raw CSV on GitHub).

Once your data is loaded, you will see a preview of the first few rows at the bottom of the page.

## 2. The Data Science Workflow

The navigation bar at the top of the page is organized to follow a logical data science workflow, from left to right.

### 2.1. Cleaning

On the **Cleaning** page, you can fix common data quality issues:
-   **Handle Missing Values:** Choose columns and a strategy (like 'drop', 'mean', 'median', or 'KNN Imputation') to handle missing data.
-   **Rename a Column:** Select a column and provide a new name.
-   **Convert Data Type:** Change a column's data type (e.g., from text to number).
-   **Handle Outliers:** Remove or cap extreme values in your numerical columns.
-   **Apply Regex:** Use regular expressions for advanced text cleaning.

### 2.2. Filtering

The **Filtering** page allows you to select a subset of your data.
-   Choose a column, an operator (like '>', 'contains', 'between'), and a value to filter your data. The available operators will change based on the type of column you select.

### 2.3. Combining

On the **Combining** page, you can merge your current dataset with a second one.
1.  **Upload Second Dataset:** Use the form to upload the second file you want to combine with.
2.  **Configure Merge:** Once the second file is uploaded, the merge controls will become active. Select the join type (inner, left, etc.) and the key columns from both tables to perform the merge.

### 2.4. Aggregation

The **Aggregation** page helps you summarize your data.
-   **Group By & Aggregate:** Group your data by one or more columns and calculate multiple summary statistics at once (e.g., the sum of sales and the average number of customers per region).
-   **Pivot Table:** Restructure your data into a pivot table to see it from a different perspective.

### 2.5. Engineering

On the **Engineering** page, you can create new features for your models.
-   **Create Feature from Columns:** Perform mathematical operations on two columns to create a new one.
-   **Apply Encoding:** Convert categorical columns into a numerical format that models can understand. Advanced methods like Target Encoding are available.
-   **Bin Numeric Column:** Group a numerical column into a smaller number of bins.
-   **Scale Features:** Standardize the scale of your numerical features.
-   **Create Polynomial Features:** Generate polynomial and interaction features.

### 2.6. EDA (Exploratory Data Analysis)

The **EDA** page is for visualizing your data.
-   **Automated Report:** Get a quick overview of your dataset's properties.
-   **Univariate, Bivariate, Multivariate Tabs:** Create a wide variety of plots to explore your data, from simple histograms to complex correlation heatmaps.

### 2.7. Modeling

The **Modeling** page is where you build machine learning models.
1.  **Select Model Type:** Choose between Classification, Regression, and Clustering.
2.  **Select Model:** Choose a specific algorithm from the list.
3.  **Select Target Column:** (For Classification/Regression) Choose the column you want to predict.
4.  **Run Model:** The app will train the model and generate a comprehensive report, including performance metrics and a SHAP plot for feature importance.

## 3. Saving Your Work

Use the **Projects** page to save your entire session, including the data and the history of all operations. You can load a project later to continue where you left off.

## 4. Exporting Your Data

At any point in the workflow, you can use the **Export** dropdown in the navigation bar to download your current dataset in CSV, Excel, JSON, or Parquet format.
