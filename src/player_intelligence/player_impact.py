"""Infer likely squads from recent IPL appearances and calculate player impact scores."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.config import PROCESSED_DIR


def likely_players(team: str, cutoff: date | None = None, limit: int = 11) -> list[dict]:
    """Return the strongest recently used IPL players, not a confirmed playing XI."""
    cutoff = pd.Timestamp(cutoff or date.today())
    players = pd.read_csv(PROCESSED_DIR / "ipl_player_match_stats.csv")
    matches = pd.read_csv(PROCESSED_DIR / "ipl_matches.csv", usecols=["match_id", "date"])
    data = players.merge(matches, on="match_id", how="inner")
    data["date"] = pd.to_datetime(data["date"])
    data = data[(data["team"] == team) & (data["date"] < cutoff)].copy()
    if data.empty:
        return []
    # Weight only the latest ten appearances per player, which makes this a current-form score.
    data = data.sort_values("date", ascending=False).groupby("player", group_keys=False).head(10)
    numeric = ["runs", "balls_faced", "balls_bowled", "runs_conceded", "wickets"]
    data[numeric] = data[numeric].fillna(0)
    grouped = data.groupby("player").agg(matches=("match_id", "nunique"), last_match=("date", "max"),
        runs=("runs", "sum"), balls_faced=("balls_faced", "sum"), balls_bowled=("balls_bowled", "sum"),
        runs_conceded=("runs_conceded", "sum"), wickets=("wickets", "sum")).reset_index()
    grouped["strike_rate"] = 100 * grouped.runs / grouped.balls_faced.clip(lower=1)
    grouped["economy"] = 6 * grouped.runs_conceded / grouped.balls_bowled.clip(lower=1)
    grouped["batting_score"] = (grouped.runs / grouped.matches.clip(lower=1) / 30 + grouped.strike_rate / 150).clip(upper=2)
    grouped["bowling_score"] = (grouped.wickets / grouped.matches.clip(lower=1) / 1.5 + (10 - grouped.economy).clip(lower=0) / 10).clip(upper=2)
    grouped["impact_score"] = (0.55 * grouped.batting_score + 0.45 * grouped.bowling_score + 0.05 * grouped.matches).round(3)
    return grouped.sort_values(["impact_score", "matches"], ascending=False).head(limit).assign(last_match=lambda x: x.last_match.dt.date.astype(str)).to_dict("records")


def team_player_impacts(team_a: str, team_b: str, cutoff: date | None = None) -> dict[str, list[dict]]:
    return {team_a: likely_players(team_a, cutoff), team_b: likely_players(team_b, cutoff)}
