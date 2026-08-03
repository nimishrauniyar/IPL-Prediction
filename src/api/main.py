"""Local API exposing the IPL prediction pipeline to the React client."""

from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd

from src.config import PROCESSED_DIR
from src.context.match_context import canonical_team
from src.models.train import MODEL_DIR
from src.insights import build_match_insights

app = FastAPI(title="IPL Match Intelligence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/teams")
def teams() -> dict:
    matches = pd.read_csv(PROCESSED_DIR / "ipl_matches.csv", usecols=["team_a", "team_b"])
    return {"teams": sorted(set(matches.team_a.dropna()) | set(matches.team_b.dropna()))}


@app.get("/api/seasons")
def seasons() -> dict:
    matches = pd.read_csv(PROCESSED_DIR / "ipl_matches.csv", usecols=["season"])
    values = sorted(matches.season.dropna().astype(str).unique(), reverse=True)
    return {"seasons": values}


@app.get("/api/fixtures")
def fixtures(season: Optional[str] = None) -> dict:
    """Return leakage-safe pre-match predictions for every fixture in a season."""
    matches = pd.read_csv(PROCESSED_DIR / "ipl_matches.csv")
    features = pd.read_csv(PROCESSED_DIR / "ipl_match_features.csv")
    available_seasons = sorted(matches.season.dropna().astype(str).unique())
    selected_season = str(season) if season else available_seasons[-1]
    if selected_season not in available_seasons:
        raise HTTPException(status_code=404, detail="That IPL season is not available locally.")

    fixture_rows = matches[matches.season.astype(str) == selected_season].copy()
    fixture_rows["stage"] = fixture_rows.get("stage", "League stage").fillna("League stage")
    fixture_rows = fixture_rows.merge(
        features.drop(columns=["date", "season", "team_a", "team_b", "venue"], errors="ignore"),
        on="match_id", how="inner",
    ).sort_values(["date", "match_id"])
    artifact = joblib.load(MODEL_DIR / "ipl_ensemble.joblib")
    x = fixture_rows[artifact["feature_columns"]]
    probabilities = sum(weight * artifact["models"][name].predict_proba(x)[:, 1]
                        for name, weight in artifact["weights"].items())
    result = []
    for row, probability in zip(fixture_rows.itertuples(index=False), probabilities):
        support = min(row.team_a_matches_played, row.team_b_matches_played)
        result.append({
            "match_id": str(row.match_id), "date": pd.Timestamp(row.date).date().isoformat(),
            "team_a": row.team_a, "team_b": row.team_b, "venue": row.venue,
            "stage": row.stage, "team_a_win_probability": round(float(probability), 4),
            "team_b_win_probability": round(float(1 - probability), 4),
            "confidence": "high" if support >= 30 else "medium" if support >= 10 else "low",
        })
    return {"season": selected_season, "fixtures": result,
            "note": "Predictions use only information available before each fixture; actual results are not shown."}


@app.get("/api/insights")
def insights(team_a: str, team_b: str) -> dict:
    if canonical_team(team_a) == canonical_team(team_b):
        raise HTTPException(status_code=400, detail="Choose two different teams.")
    try:
        return build_match_insights(team_a, team_b)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@app.get("/api/h2h")
def h2h(team_a: str, team_b: str) -> dict:
    matches = pd.read_csv(PROCESSED_DIR / "ipl_matches.csv")
    
    # Calculate H2H
    h2h_matches = matches[
        ((matches.team_a == team_a) & (matches.team_b == team_b)) | 
        ((matches.team_a == team_b) & (matches.team_b == team_a))
    ]
    a_wins = len(h2h_matches[h2h_matches.winner == team_a])
    b_wins = len(h2h_matches[h2h_matches.winner == team_b])
    total = len(h2h_matches)

    # Calculate Form
    def get_form(team, limit=5):
        team_matches = matches[(matches.team_a == team) | (matches.team_b == team)].sort_values('date', ascending=False).head(limit)
        form = []
        for _, row in team_matches.iterrows():
            if pd.isna(row.winner):
                form.append("NR")
            elif row.winner == team:
                form.append("W")
            else:
                form.append("L")
        return form[::-1]

    return {
        "team_a": team_a,
        "team_b": team_b,
        "h2h": {
            "matches": total,
            "team_a_wins": a_wins,
            "team_b_wins": b_wins,
        },
        "form_a": get_form(team_a),
        "form_b": get_form(team_b)
    }
