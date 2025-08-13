import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix
)
import plotly.express as px
import plotly.figure_factory as ff

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

def get_model_list():
    """
    Returns a dictionary of available classification and regression models.
    """
    models = {
        "Classification": {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Support Vector Machine": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Extra Trees": ExtraTreesClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "LightGBM": LGBMClassifier(),
            "CatBoost": CatBoostClassifier(verbose=0),
            "Gaussian Naive Bayes": GaussianNB(),
            "SGD Classifier": SGDClassifier(),
        },
        "Regression": {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "ElasticNet": ElasticNet(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Support Vector Regressor": SVR(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Extra Trees Regressor": ExtraTreesRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "XGBoost Regressor": XGBRegressor(eval_metric='rmse'),
            "LightGBM Regressor": LGBMRegressor(),
            "CatBoost Regressor": CatBoostRegressor(verbose=0),
            "Bayesian Ridge Regression": BayesianRidge(),
        }
    }
    return models


def run_models(df, target_column, model_type, model_name):
    """
    Runs a specified machine learning model on the preprocessed data.

    Technical: This function is the core of the modeling process. It separates
    the target variable from the features, splits the data into training and
    testing sets, trains the selected model, and then makes predictions.
    It returns a comprehensive report of the model's performance metrics.

    Layman: This is where the magic happens. You choose a type of model
    (like a "decision tree") and tell it which column you want to predict.
    The function trains the model and then tells you how well it performed
    on new, unseen data, giving you scores like accuracy.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        target_column (str): The name of the column to predict.
        model_type (str): The type of model to run ('Classification' or 'Regression').
        model_name (str): The name of the specific model to use.

    Returns:
        dict: A dictionary containing the model's performance report.
    """
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    # Drop non-numeric and target columns from features
    features = df.drop(columns=[target_column]).select_dtypes(include=np.number).columns
    if not list(features):
        return {"error": "No numeric features found for modeling."}

    X = df[features]
    y = df[target_column]
    
    # Check for non-numeric target column in Classification
    if model_type == 'Classification' and not pd.api.types.is_numeric_dtype(y):
        # Apply Label Encoding to the target column
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        models = get_model_list()
        model = models[model_type][model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = {
            "model_name": model_name,
            "model_type": model_type,
            "metrics": {}
        }

        if model_type == 'Classification':
            report['metrics']['Accuracy'] = accuracy_score(y_test, y_pred)
            report['metrics']['Precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            report['metrics']['Recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            report['metrics']['F1 Score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    report['metrics']['ROC AUC'] = roc_auc_score(y_test, y_proba[:, 1])
            
            # Generate Confusion Matrix plot as a JSON string
            cm = confusion_matrix(y_test, y_pred)
            cm_text = [[str(y) for y in x] for x in cm]
            fig = ff.create_annotated_heatmap(cm, x=le.classes_.tolist() if model_type == 'Classification' and 'le' in locals() else None, y=le.classes_.tolist() if model_type == 'Classification' and 'le' in locals() else None, annotation_text=cm_text, colorscale='Viridis')
            fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
            report['confusion_matrix'] = fig.to_json()
            
        elif model_type == 'Regression':
            report['metrics']['MSE'] = mean_squared_error(y_test, y_pred)
            report['metrics']['MAE'] = mean_absolute_error(y_test, y_pred)
            report['metrics']['R2 Score'] = r2_score(y_test, y_pred)
            
            # Generate Regression plot
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title=f'{model_name} Regression')
            fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Fit'))
            report['regression_plot'] = fig.to_json()

        return report
    except KeyError:
        return {"error": f"Model '{model_name}' not found for type '{model_type}'."}
    except Exception as e:
        return {"error": f"An unexpected error occurred during modeling: {e}"}
