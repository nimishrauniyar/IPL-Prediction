"""Train Phase 4 baselines and the calibrated Phase 5 XGBoost ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
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


def model_definitions() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42))]),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features=0.5, random_state=42, class_weight="balanced"),
        "xgboost": CalibratedClassifierCV(
            estimator=XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.7,
                                    colsample_bytree=0.9, reg_alpha=0.5, reg_lambda=2.0, eval_metric="logloss", random_state=42),
            method="sigmoid", cv=TimeSeriesSplit(n_splits=5),
        ),
    }


def fit_and_evaluate(features: pd.DataFrame | None = None) -> dict:
    features = features if features is not None else pd.read_csv(PROCESSED_DIR / "ipl_match_features.csv")
    features["year"] = pd.to_datetime(features["date"]).dt.year
    train = features[features.year <= 2024]
    validation = features[features.year == 2025]
    test = features[features.year == 2026]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Expected IPL seasons through 2026 for train/validation/test splits.")
    x_train, y_train = train[FEATURE_COLUMNS], train.target_team_a_win
    results, fitted = {}, {}
    validation_predictions = {}
    for name, model in model_definitions().items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        results[name] = {"validation_2025": metrics(validation.target_team_a_win, probabilities)}
        validation_predictions[name] = probabilities
        fitted[name] = model
    # Refit the final models through 2025, then report their untouched 2026 performance.
    development = features[features.year <= 2025]
    final_models, test_predictions = {}, {}
    for name, model in model_definitions().items():
        model.fit(development[FEATURE_COLUMNS], development.target_team_a_win)
        probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        results[name]["test_2026"] = metrics(test.target_team_a_win, probabilities)
        final_models[name], test_predictions[name] = model, probabilities

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
