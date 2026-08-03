"""Train Phase 4 baselines and the calibrated Phase 5 XGBoost ensemble."""

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
    validation_predictions = {}
    
    for name, config in model_definitions().items():
        print(f"Tuning {name}...")
        base_model = config["model"]
        grid = config["grid"]
        wp = config["weight_param"]
        
        # Hyperparameter Tuning with TimeSeriesSplit to prevent leakage
        search = RandomizedSearchCV(base_model, grid, n_iter=8, scoring="roc_auc", 
                                    cv=TimeSeriesSplit(n_splits=3), random_state=42, n_jobs=-1)
        fit_params = {wp: w_train} if wp else {}
        search.fit(x_train, y_train, **fit_params)
        best_model = search.best_estimator_
        
        print(f"Best params for {name}: {search.best_params_}")
        
        probabilities = best_model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        results[name] = {"validation_2025": metrics(validation.target_team_a_win, probabilities)}
        validation_predictions[name] = probabilities
        fitted[name] = best_model

    # Refit the final models through 2025, then report their untouched 2026 performance.
    development = features[features.year <= 2025]
    w_dev = weights_array[features.year <= 2025]
    final_models, test_predictions = {}, {}
    
    for name, config in model_definitions().items():
        best_model = fitted[name] # Reuse the tuned model architecture
        wp = config["weight_param"]
        fit_params = {wp: w_dev} if wp else {}
        best_model.fit(development[FEATURE_COLUMNS], development.target_team_a_win, **fit_params)
        probabilities = best_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        results[name]["test_2026"] = metrics(test.target_team_a_win, probabilities)
        final_models[name], test_predictions[name] = best_model, probabilities

    # Dynamic weights based on validation AUC
    weights = {}
    total_weight = 0.0
    for name in final_models.keys():
        auc = results[name]["validation_2025"]["roc_auc"]
        weight = max(0.0, auc - 0.5) ** 2  # Square to emphasize better models
        weights[name] = weight
        total_weight += weight
        
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        weights = {"logistic_regression": 1.0, "random_forest": 0.0, "xgboost": 0.0}

    # Ensemble test predictions
    ensemble_preds = sum(test_predictions[name] * weights[name] for name in final_models.keys())
    results["ensemble_model"] = {"test_2026": metrics(test.target_team_a_win, ensemble_preds), "weights": weights}
    
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"models": final_models, "feature_columns": FEATURE_COLUMNS,
                 "weights": weights},
                MODEL_DIR / "ipl_ensemble.joblib")
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    print(json.dumps(fit_and_evaluate(), indent=2))
