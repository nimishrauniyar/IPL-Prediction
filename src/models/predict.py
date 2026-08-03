"""Make a local pre-match prediction from two teams only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date

import joblib
import pandas as pd

from src.config import PROCESSED_DIR
from src.context.match_context import get_match_context
from src.features.build_features import build_state_before, make_feature_row
from src.models.train import MODEL_DIR


def predict_match(team_a: str, team_b: str, schedule_path=None, as_of: date | None = None) -> dict:
    """Return ensemble probabilities, using only data available before the fixture date."""
    context = get_match_context(team_a, team_b, schedule_path=schedule_path, today=as_of)
    cutoff = pd.Timestamp(context.match_date or as_of or date.today())
    matches = pd.read_csv(PROCESSED_DIR / "ipl_matches.csv")
    team_stats = pd.read_csv(PROCESSED_DIR / "ipl_team_match_stats.csv")
    state = build_state_before(matches, team_stats, cutoff)
    venue = context.venue if context.mode == "scheduled_match" else None
    feature_row = make_feature_row(state, context.team_a, context.team_b, venue)
    artifact = joblib.load(MODEL_DIR / "ipl_ensemble.joblib")
    x = pd.DataFrame([feature_row], columns=artifact["feature_columns"])
    probability = sum(artifact["weights"][name] * model.predict_proba(x)[:, 1][0]
                      for name, model in artifact["models"].items())

    # Confidence describes how much IPL history supports the estimate, not how
    # lopsided the probability is.  A well-supported 50/50 matchup should not
    # be labelled low confidence merely because it is genuinely close.
    prior_matches = min(feature_row["team_a_matches_played"], feature_row["team_b_matches_played"])
    confidence = "high" if prior_matches >= 30 else "medium" if prior_matches >= 10 else "low"
    return {**asdict(context), "team_a_win_probability": round(float(probability), 4),
            "team_b_win_probability": round(float(1 - probability), 4), "confidence": confidence,
            "features_used": feature_row}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict an IPL matchup from two team names.")
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    args = parser.parse_args()
    print(predict_match(args.team_a, args.team_b))
