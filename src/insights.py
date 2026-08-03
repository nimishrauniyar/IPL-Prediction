"""One command that produces prediction, explanation, player impact, and grounded preview."""

from __future__ import annotations

import argparse
import json

from src.explainability.shap_explainer import explain_xgboost
from src.models.predict import predict_match
from src.player_intelligence.player_impact import team_player_impacts
from src.rag.narrative import generate_grounded_preview


def build_match_insights(team_a: str, team_b: str) -> dict:
    prediction = predict_match(team_a, team_b)
    return {
        "prediction": {key: value for key, value in prediction.items() if key != "features_used"},
        "model_explanation": explain_xgboost(prediction["features_used"]),
        "inferred_player_impacts": team_player_impacts(prediction["team_a"], prediction["team_b"], prediction.get("match_date")),
        "grounded_preview": generate_grounded_preview(prediction),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create full IPL team-only match insights.")
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    args = parser.parse_args()
    print(json.dumps(build_match_insights(args.team_a, args.team_b), indent=2, default=str))
