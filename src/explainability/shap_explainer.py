"""Explain the XGBoost component of an ensemble prediction with SHAP."""

from __future__ import annotations

import joblib
import pandas as pd

from src.models.train import MODEL_DIR

DISPLAY_NAMES = {
    "elo_difference": "team strength (Elo)", "recent_win_rate_difference": "recent win rate",
    "overall_win_rate_difference": "overall win rate", "recent_run_rate_difference": "recent batting run rate",
    "recent_concede_rate_difference": "recent bowling/conceding rate", "head_to_head_advantage": "head-to-head record",
    "venue_win_rate_difference": "venue record", "team_a_matches_played": "Team A IPL experience",
    "team_b_matches_played": "Team B IPL experience",
    "h2h_recent_momentum": "head-to-head momentum",
    "venue_dominance_score_diff": "venue dominance",
    "elo_momentum_diff": "momentum (Elo trend)",
    "win_streak_diff": "recent win streak",
    "net_run_rate_diff": "net run rate",
    "batting_vs_bowling_matchup_a": "Team A batting vs B bowling",
    "batting_vs_bowling_matchup_b": "Team B batting vs A bowling",
}


def explain_xgboost(feature_row: dict[str, float], limit: int = 5) -> list[dict]:
    """Return the strongest local factors from the deployed logistic model."""
    artifact = joblib.load(MODEL_DIR / "ipl_ensemble.joblib")
    x = pd.DataFrame([feature_row], columns=artifact["feature_columns"])
    pipeline = artifact["models"]["logistic_regression"]
    scaled = pipeline.named_steps["scale"].transform(x)
    values = scaled.reshape(-1) * pipeline.named_steps["model"].coef_.reshape(-1)
    ranked = sorted(zip(artifact["feature_columns"], values), key=lambda item: abs(item[1]), reverse=True)[:limit]
    return [{"feature": feature, "label": DISPLAY_NAMES[feature], "shap_value": round(float(value), 4),
             "favors": "Team A" if value >= 0 else "Team B", "value": round(float(feature_row[feature]), 4)}
            for feature, value in ranked]
