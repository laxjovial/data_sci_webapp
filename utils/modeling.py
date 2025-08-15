import pandas as pd
import numpy as np
import shap
import base64
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, silhouette_score
)
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder

# Classification Models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import BayesianRidge

# Clustering Models
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


def get_model_list():
    """Returns a dictionary of available models."""
    models = {
        "Classification": {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": LGBMClassifier(),
            "CatBoost": CatBoostClassifier(verbose=0),
            "Support Vector Machine": SVC(probability=True),
        },
        "Regression": {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "XGBoost Regressor": XGBRegressor(eval_metric='rmse'),
            "LightGBM Regressor": LGBMRegressor(),
            "CatBoost Regressor": CatBoostRegressor(verbose=0),
            "Support Vector Regressor": SVR(),
        },
        "Clustering": {
            "K-Means": KMeans(n_init='auto'),
            "Agglomerative Clustering": AgglomerativeClustering(),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        }
    }
    return models

def generate_shap_summary_plot(model, X_train):
    """Generates a SHAP summary plot as a base64 encoded image."""
    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        plt.figure()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_base64}" class="img-fluid"/>'
    except Exception as e:
        return f"<p>Could not generate SHAP plot: {e}</p>"


def run_model(df, target_column, model_type, model_name, model_params={}):
    """
    Runs a specified machine learning model on the data.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        target_column (str): The name of the column to predict or use for clustering.
        model_type (str): The type of model ('Classification', 'Regression', 'Clustering').
        model_name (str): The name of the specific model to use.
        model_params (dict): Parameters for the model.

    Returns:
        dict: A dictionary containing the model's performance report.
    """
    if model_type != 'Clustering' and target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    if target_column in numeric_features and model_type != 'Clustering':
        numeric_features.remove(target_column)

    if not numeric_features:
        return {"error": "No numeric features found for modeling."}

    X = df[numeric_features]
    
    try:
        models = get_model_list()
        model = models[model_type][model_name]

        # Set model parameters if provided
        if model_params:
            model.set_params(**model_params)

        report = {"model_name": model_name, "model_type": model_type, "metrics": {}}

        if model_type in ['Classification', 'Regression']:
            y = df[target_column]
            if model_type == 'Classification' and not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Generate SHAP plot
            report['shap_plot'] = generate_shap_summary_plot(model, X_train)

            if model_type == 'Classification':
                report['metrics']['Accuracy'] = accuracy_score(y_test, y_pred)
                report['metrics']['F1 Score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                fig = ff.create_annotated_heatmap(cm, colorscale='Viridis')
                report['confusion_matrix'] = fig.to_json()
            else: # Regression
                report['metrics']['R2 Score'] = r2_score(y_test, y_pred)
                report['metrics']['Mean Squared Error'] = mean_squared_error(y_test, y_pred)

        elif model_type == 'Clustering':
            model.fit(X)
            labels = model.labels_
            df['cluster'] = labels
            report['metrics']['Number of Clusters'] = len(np.unique(labels))
            if len(np.unique(labels)) > 1:
                report['metrics']['Silhouette Score'] = silhouette_score(X, labels)
            
            # Create a 2D scatter plot of the clusters
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            plot_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
            plot_df['cluster'] = labels
            fig = px.scatter(plot_df, x='PCA1', y='PCA2', color='cluster', title='Clusters (via PCA)')
            report['cluster_plot'] = fig.to_json()

        return report

    except KeyError:
        return {"error": f"Model '{model_name}' not found for type '{model_type}'."}
    except Exception as e:
        return {"error": f"An unexpected error occurred during modeling: {e}"}
