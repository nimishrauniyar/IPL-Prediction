"""Train Phase 4 baselines and the calibrated Phase 5 XGBoost ensemble with Meta-Stacking."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import PROJECT_ROOT, PROCESSED_DIR
from src.features.build_features import FEATURE_COLUMNS

MODEL_DIR = PROJECT_ROOT / "models"


def metrics(y_true, probabilities) -> dict[str, float]:
    return {"accuracy": round(accuracy_score(y_true, probabilities >= 0.5), 4),
            "roc_auc": round(roc_auc_score(y_true, probabilities), 4),
            "log_loss": round(log_loss(y_true, probabilities, labels=[0, 1]), 4),
            "brier_score": round(brier_score_loss(y_true, probabilities), 4)}


def model_definitions() -> dict[str, dict]:
    return {
        "logistic_regression": {
            "model": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=42))]),
            "grid": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
            "weight_param": "clf__sample_weight"
        },
        "random_forest": {
            "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "grid": {"n_estimators": [100, 300, 500], "max_depth": [5, 10, 15, None], "min_samples_leaf": [2, 5, 10]},
            "weight_param": "sample_weight"
        },
        "xgboost": {
            "model": XGBClassifier(eval_metric="logloss", random_state=42),
            "grid": {"n_estimators": [100, 200, 300], "max_depth": [2, 3, 5], "learning_rate": [0.01, 0.05, 0.1]},
            "weight_param": "sample_weight"
        },
    }


def fit_and_evaluate(features: pd.DataFrame | None = None) -> dict:
    features = features if features is not None else pd.read_csv(PROCESSED_DIR / "ipl_match_features.csv")
    features["year"] = pd.to_datetime(features["date"]).dt.year
    
    # Exponential Time-Decay: More recent matches are weighted closer to 1.0
    weights_array = np.exp((features["year"] - 2026) * 0.15).values

    train = features[features.year <= 2024]
    validation = features[features.year == 2025]
    test = features[features.year == 2026]
    
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Expected IPL seasons through 2026 for train/validation/test splits.")
        
    x_train, y_train = train[FEATURE_COLUMNS], train.target_team_a_win
    w_train = weights_array[features.year <= 2024]
    
    results, fitted = {}, {}
    
    # 1. Train Base Models up to 2024
    for name, config in model_definitions().items():
        print(f"Tuning {name}...")
        base_model = config["model"]
        grid = config["grid"]
        wp = config["weight_param"]
        
        search = RandomizedSearchCV(base_model, grid, n_iter=8, scoring="roc_auc", 
                                    cv=TimeSeriesSplit(n_splits=3), random_state=42, n_jobs=-1)
        fit_params = {wp: w_train} if wp else {}
        search.fit(x_train, y_train, **fit_params)
        fitted[name] = search.best_estimator_

    # 2. Generate Meta-Features using Validation Set (2025)
    # The Meta-Model will learn how to combine the base models based on their 2025 performance
    x_val, y_val = validation[FEATURE_COLUMNS], validation.target_team_a_win
    w_val = weights_array[features.year == 2025]
    
    meta_features_val = {}
    for name, model in fitted.items():
        meta_features_val[name] = model.predict_proba(x_val)[:, 1]
    
    X_meta_train = pd.DataFrame(meta_features_val)
    
    # 3. Train the Meta-Model (Logistic Regression Stacker)
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(X_meta_train, y_val, sample_weight=w_val)
    
    print("Meta-Model Weights:", dict(zip(X_meta_train.columns, meta_model.coef_[0])))

    # 4. Refit base models on all data up to 2025 for final deployment
    development = features[features.year <= 2025]
    w_dev = weights_array[features.year <= 2025]
    final_models = {}
    
    for name, config in model_definitions().items():
        best_model = fitted[name] # Reuse tuned architecture
        wp = config["weight_param"]
        fit_params = {wp: w_dev} if wp else {}
        best_model.fit(development[FEATURE_COLUMNS], development.target_team_a_win, **fit_params)
        final_models[name] = best_model

    # 5. Evaluate on Test Set (2026)
    x_test, y_test = test[FEATURE_COLUMNS], test.target_team_a_win
    meta_features_test = {}
    
    for name, model in final_models.items():
        probabilities = model.predict_proba(x_test)[:, 1]
        meta_features_test[name] = probabilities
        results[name] = {"test_2026": metrics(y_test, probabilities)}
        
    X_meta_test = pd.DataFrame(meta_features_test)
    ensemble_preds = meta_model.predict_proba(X_meta_test)[:, 1]
    
    results["ensemble_model"] = {
        "test_2026": metrics(y_test, ensemble_preds), 
        "meta_weights": dict(zip(X_meta_test.columns, [float(x) for x in meta_model.coef_[0]]))
    }
    
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "models": final_models, 
        "meta_model": meta_model,
        "feature_columns": FEATURE_COLUMNS,
    }, MODEL_DIR / "ipl_ensemble.joblib")
    
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    print(json.dumps(fit_and_evaluate(), indent=2))
