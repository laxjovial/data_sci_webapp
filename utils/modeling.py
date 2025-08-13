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

        metrics = {"Model": name}
        if problem_type == "Classification":
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            metrics.update({
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1-Score": f1_score(y_test, y_pred, average='weighted'),
                "ROC AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A",
                "Confusion Matrix": create_confusion_matrix_plot(y_test, y_pred, model.classes_)
            })
        else: # Regression
            metrics.update({
                "R-squared": r2_score(y_test, y_pred),
                "Mean Squared Error (MSE)": mean_squared_error(y_test, y_pred),
                "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_pred),
            })

        # Add feature importance plot for all models
        metrics["Feature Importance"] = generate_feature_importance_plot(model, features)
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

def generate_feature_importance_plot(model, feature_names):
    """
    Generates a Plotly feature importance plot for a given trained model.
    """
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, use the coefficients
        importances = np.abs(model.coef_[0])

    if importances is None:
        return None

    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=True)

    fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Feature Importance')
    fig.update_layout(yaxis_title="Feature", xaxis_title="Importance Score")
    return fig.to_json()

def get_hyperparameter_grid():
    """
    Returns a dictionary of hyperparameters for tuning each model.
    This provides a starting point for users.
    """
    grids = {
        "Classification": {
            "Logistic Regression": {
                'C': [0.1, 1, 10],
                'solver': ['liblinear']
            },
            "K-Nearest Neighbors": {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            "Support Vector Machine": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            },
            "Decision Tree": {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            "Random Forest": {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20]
            },
            "AdaBoost": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 1]
            },
            "Gradient Boosting": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            "XGBoost": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        "Regression": {
            # Add regression model grids here if needed
        }
    }
    return grids

def tune_model_hyperparameters(df, features, target, problem_type, model_name, param_grid, test_size=0.3, random_state=42):
    """
    Tunes hyperparameters for a single model using GridSearchCV.
    """
    from sklearn.model_selection import GridSearchCV

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = get_model_list()[problem_type][model_name]

    # Define scoring metric
    scoring = 'f1_weighted' if problem_type == "Classification" else 'r2'

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring=scoring)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    # Evaluate the best model found by grid search on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    if problem_type == "Classification":
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
        final_metrics = {
            "Best Parameters": str(best_params),
            "CV Score": grid_search.best_score_,
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Test F1-Score": f1_score(y_test, y_pred, average='weighted'),
            "Test ROC AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
        }
    else: # Regression
        final_metrics = {
            "Best Parameters": str(best_params),
            "CV Score (R-squared)": grid_search.best_score_,
            "Test R-squared": r2_score(y_test, y_pred),
            "Test MSE": mean_squared_error(y_test, y_pred)
        }

    return final_metrics
