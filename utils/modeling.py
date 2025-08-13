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
            "Bayesian Ridge": BayesianRidge(),
        }
    }
    return models

def run_models(df, features, target, problem_type, test_size=0.3, random_state=42):
    """
    Splits data, trains a suite of models, and evaluates them.
    """
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = get_model_list()[problem_type]
    results = []

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {"Model": name}
            if problem_type == "Classification":
                y_prob = None
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                
                metrics.update({
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    "ROC AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A",
                    "Confusion Matrix": create_confusion_matrix_plot(y_test, y_pred, model.classes_ if hasattr(model, 'classes_') else np.unique(y_test))
                })
            else: # Regression
                metrics.update({
                    "R-squared": r2_score(y_test, y_pred),
                    "Mean Squared Error (MSE)": mean_squared_error(y_test, y_pred),
                    "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_pred),
                })
            
            metrics["Feature Importance"] = generate_feature_importance_plot(model, features)
            results.append(metrics)
        except Exception as e:
            results.append({"Model": name, "Error": str(e)})

    return pd.DataFrame(results)

def create_confusion_matrix_plot(y_true, y_pred, labels):
    """
    Generates a Plotly confusion matrix figure.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    z_text = [[str(y) for y in x] for x in cm]
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=list(labels),
        y=list(labels),
        annotation_text=z_text,
        colorscale='Viridis'
    )
    fig.update_layout(title_text='Confusion Matrix')
    return fig.to_json()

def generate_feature_importance_plot(model, feature_names):
    """
    Generates a Plotly feature importance plot for a given trained model.
    """
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, use the absolute value of the coefficients
        coef = model.coef_
        if len(coef.shape) > 1: # Multi-class
            importances = np.abs(coef).mean(axis=0)
        else: # Binary or regression
            importances = np.abs(coef)
    
    if importances is None or len(importances) == 0:
        return None

    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=True)
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Feature Importance')
    fig.update_layout(yaxis_title="Feature", xaxis_title="Importance Score")
    return fig.to_json()
