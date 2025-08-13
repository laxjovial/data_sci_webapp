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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def get_model_list():
    """
    Returns a dictionary of available classification and regression models.
    """
    models = {
        "Classification": {
            "Logistic Regression": LogisticRegression(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Support Vector Machine": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        },
        "Regression": {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Support Vector Regressor": SVR(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "XGBoost Regressor": XGBRegressor(eval_metric='rmse')
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
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if problem_type == "Classification":
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1-Score": f1_score(y_test, y_pred, average='weighted'),
                "ROC AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A",
                "Confusion Matrix": create_confusion_matrix_plot(y_test, y_pred, model.classes_)
            }
        else: # Regression
            metrics = {
                "Model": name,
                "R-squared": r2_score(y_test, y_pred),
                "Mean Squared Error (MSE)": mean_squared_error(y_test, y_pred),
                "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_pred),
            }
        results.append(metrics)

    return pd.DataFrame(results)

def create_confusion_matrix_plot(y_true, y_pred, labels):
    """
    Generates a Plotly confusion matrix figure.
    """
    cm = confusion_matrix(y_true, y_pred)
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
