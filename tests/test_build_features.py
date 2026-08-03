import pandas as pd

from src.features.build_features import FEATURE_COLUMNS, build_feature_table


def test_features_do_not_leak_current_match_outcome():
    matches = pd.DataFrame([
        {"match_id": "1", "date": "2024-01-01", "team_a": "A", "team_b": "B", "venue": "V", "winner": "A", "season": "2024"},
        {"match_id": "2", "date": "2024-01-02", "team_a": "A", "team_b": "B", "venue": "V", "winner": "B", "season": "2024"},
    ])
    stats = pd.DataFrame([
        {"match_id": "1", "team": "A", "runs_scored": 150, "balls_faced": 120}, {"match_id": "1", "team": "B", "runs_scored": 140, "balls_faced": 120},
        {"match_id": "2", "team": "A", "runs_scored": 130, "balls_faced": 120}, {"match_id": "2", "team": "B", "runs_scored": 131, "balls_faced": 120},
    ])
    features = build_feature_table(matches, stats)
    assert features.loc[0, "recent_win_rate_difference"] == 0
    assert features.loc[1, "recent_win_rate_difference"] == 1
    assert set(FEATURE_COLUMNS).issubset(features.columns)
