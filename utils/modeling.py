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
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
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
            "Extra Trees": {
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
            },
            "LightGBM": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50]
            },
            "CatBoost": {
                'iterations': [100, 200],
                'learning_rate': [0.01, 0.1],
                'depth': [4, 6]
            },
            "SGD Classifier": {
                'alpha': [0.0001, 0.001, 0.01],
                'loss': ['hinge', 'log_loss']
            }
        },
        "Regression": {
            "Ridge Regression": {'alpha': [0.1, 1.0, 10.0]},
            "Lasso Regression": {'alpha': [0.1, 1.0, 10.0]},
            "ElasticNet": {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8]},
            "Random Forest Regressor": {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
            "Extra Trees Regressor": {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        }
    }
    return grids

def tune_model_hyperparameters(df, features, target, problem_type, model_name, param_grid, method='grid', test_size=0.3, random_state=42):
    """
    Tunes hyperparameters for a single model using various search strategies.
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = get_model_list()[problem_type][model_name]
    scoring = 'f1_weighted' if problem_type == "Classification" else 'r2'

    search_cv = None
    if method == 'grid':
        search_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring=scoring)
    elif method == 'random':
        # n_iter controls how many random combinations are tried
        search_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=20, cv=3, n_jobs=-1, verbose=2, scoring=scoring, random_state=random_state)
    elif method == 'bayesian':
        # BayesSearchCV requires a different format for the search space
        search_spaces = {}
        for param, values in param_grid.items():
            if isinstance(values[0], str):
                search_spaces[param] = Categorical(values)
            elif isinstance(values[0], int) and len(values) > 1 and values[1] > values[0]:
                 search_spaces[param] = Integer(values[0], values[-1])
            elif isinstance(values[0], float) and len(values) > 1 and values[1] > values[0]:
                 search_spaces[param] = Real(values[0], values[-1])
            else: # Fallback for single values or complex types
                search_spaces[param] = Categorical(values)

        search_cv = BayesSearchCV(estimator=model, search_spaces=search_spaces, n_iter=32, cv=3, n_jobs=-1, verbose=2, scoring=scoring, random_state=random_state)

    if search_cv:
        search_cv.fit(X_train, y_train)
        best_params = search_cv.best_params_
        best_model = search_cv.best_estimator_
        y_pred = best_model.predict(X_test)

        if problem_type == "Classification":
            y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
            final_metrics = {
                "Best Parameters": str(best_params),
                "CV Score": search_cv.best_score_,
                "Test Accuracy": accuracy_score(y_test, y_pred),
                "Test F1-Score": f1_score(y_test, y_pred, average='weighted'),
                "Test ROC AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
            }
        else: # Regression
            final_metrics = {
                "Best Parameters": str(best_params),
                "CV Score (R-squared)": search_cv.best_score_,
                "Test R-squared": r2_score(y_test, y_pred),
                "Test MSE": mean_squared_error(y_test, y_pred)
            }
        return final_metrics

    return {"Error": "Invalid tuning method specified."}
