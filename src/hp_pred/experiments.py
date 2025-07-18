from pathlib import Path
import multiprocessing as mp
import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import auc, roc_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from aeon.classification.sklearn import RotationForestClassifier
from sklearn.metrics import roc_auc_score

from optuna import Trial


NUMBER_CV_FOLD = 3
N_INTERPOLATION = 1000

def objective_rotationforest(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC score for RotationForest model using Optuna hyperparameter optimization.
    
    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
        feature_name (List[str]): List of feature names to use for training.
    Returns:
        float: Mean AUC score of the RotationForest model.
    
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_group": trial.suggest_int("min_group", 3, 3),  # Fixed value of 3 to avoid errors
        "max_group": trial.suggest_int("max_group", 3, 3),  # Fixed value of 3 to avoid errors
        "remove_proportion": trial.suggest_float("remove_proportion", 0.0, 0.9),
        "pca_solver": trial.suggest_categorical("pca_solver", ["auto", "full", "randomized"]),
        "n_jobs": os.cpu_count(),
    }
    
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."
    
    fold_number = 0
    # separate training in folds
    ap_scores = np.zeros(number_cv_fold)
    for i in range(number_cv_fold):
        
        X_train = data_train[i][feature_name]
        y_train = data_train[i].label
        
        X_validate = data_test[i][feature_name]
        y_validate = data_test[i].label
        
        optuna_model = RotationForestClassifier(**params)
        optuna_model.fit(X_train, y_train)
        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]
        
        # Evaluate predictions with AP score
        ap_scores[fold_number] = average_precision_score(y_validate, y_pred)
        fold_number += 1
    
    return ap_scores.mean()

def objective_randomforest(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC score for RandomForest model using Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
        feature_name (List[str]): List of feature names to use for training.
    Returns:
        float: Mean AUC score of the RandomForest model.

    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 1, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "n_jobs": os.cpu_count(),
    }
    
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."

    fold_number = 0
    # separate training in folds
    ap_scores = np.zeros(number_cv_fold)
    for i in range(number_cv_fold):

        X_train = data_train[i][feature_name]
        y_train = data_train[i].label

        X_validate = data_test[i][feature_name]
        y_validate = data_test[i].label

        optuna_model = RandomForestClassifier(**params)
        optuna_model.fit(X_train, y_train)
        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]

        # Evaluate predictions with AP score
        ap_scores[fold_number] = average_precision_score(y_validate, y_pred)
        fold_number += 1

    return ap_scores.mean()

def objective_logistic_regression(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC score for Logistic Regression model using Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
        feature_name (List[str]): List of feature names to use for training.
    Returns:
        float: Mean AUC score of the Logistic Regression model.

    """
    params = {
        "C": trial.suggest_float("C", 1e-5, 10.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None]),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
        "max_iter": trial.suggest_int("max_iter", 1000, 10000),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "random_state": 42,
        "n_jobs": os.cpu_count(),
    }
    
    # Ensure compatible solver and penalty combinations
    if params["penalty"] == "elasticnet" and params["solver"] != "saga":
        params["solver"] = "saga"
    elif params["penalty"] == "l1" and params["solver"] not in ["liblinear", "saga"]:
        params["solver"] = "saga"
    elif params["penalty"] is None and params["solver"] == "liblinear":
        params["solver"] = "lbfgs"
    
    # Add l1_ratio only when elasticnet is selected
    if params["penalty"] == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."

    fold_number = 0
    # separate training in folds
    ap_scores = np.zeros(number_cv_fold)
    for i in range(number_cv_fold):

        X_train = data_train[i][feature_name]
        y_train = data_train[i].label

        X_validate = data_test[i][feature_name]
        y_validate = data_test[i].label

        optuna_model = LogisticRegression(**params)
        optuna_model.fit(X_train, y_train)
        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]

        # Evaluate predictions with AP score
        ap_scores[fold_number] = average_precision_score(y_validate, y_pred)
        fold_number += 1

    return ap_scores.mean()

def objective_xgboost_roc(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC ROC score for XGBoost model using Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
    Returns:
        float: Mean AUC ROC score of the XGBoost model.

    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.01, 1.0, log=True
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "eval_metric": trial.suggest_categorical("eval_metric", ["auc", "aucpr", "logloss", "map"]),
        "objective": "binary:logistic",
        "nthread": os.cpu_count(),
        # "scale_pos_weight": data.label.value_counts()[0] / data.label.value_counts()[1],
    }
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."

    fold_number = 0
    # separate training in 3 folds
    auc_roc_scores = np.zeros(number_cv_fold)
    for i in range(number_cv_fold):

        X_train = data_train[i][feature_name]
        y_train = data_train[i].label

        X_validate = data_test[i][feature_name]
        y_validate = data_test[i].label

        optuna_model = XGBClassifier(**params)
        optuna_model.fit(X_train, y_train)
        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]

        # Evaluate predictions with AUC ROC score
        auc_roc_scores[fold_number] = roc_auc_score(y_validate, y_pred)
        fold_number += 1

    return auc_roc_scores.mean()


def objective_xgboost(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC score for XGBoost model using Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
    Returns:
        float: Mean AUC score of the XGBoost model.

    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.01, 1.0, log=True
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "eval_metric": trial.suggest_categorical("eval_metric", ["auc", "aucpr", "logloss", "map"]),
        "objective": "binary:logistic",
        "nthread": os.cpu_count(),
        # "scale_pos_weight": data.label.value_counts()[0] / data.label.value_counts()[1],
    }
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."

    fold_number = 0
    # separate training in 3 folds
    ap_scores = np.zeros(number_cv_fold)
    for i in range(number_cv_fold):

        X_train = data_train[i][feature_name]
        y_train = data_train[i].label

        X_validate = data_test[i][feature_name]
        y_validate = data_test[i].label
        

        optuna_model = XGBClassifier(**params)
        optuna_model.fit(X_train, y_train)
        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]

        # Evaluate predictions with AP score
        ap_scores[fold_number] = average_precision_score(y_validate, y_pred)
        fold_number += 1

    return ap_scores.mean()
