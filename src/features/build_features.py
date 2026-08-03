"""Build pre-match IPL features using only information known before each match."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR

FEATURE_COLUMNS = [
    "elo_difference", "recent_win_rate_difference", "overall_win_rate_difference",
    "recent_run_rate_difference", "recent_concede_rate_difference",
    "head_to_head_advantage", "venue_win_rate_difference", "team_a_matches_played",
    "team_b_matches_played", "h2h_recent_momentum", "venue_dominance_score_diff",
    "elo_momentum_diff", "win_streak_diff", "net_run_rate_diff",
    "batting_vs_bowling_matchup_a", "batting_vs_bowling_matchup_b"
]


def _rate(values: list[float], default: float = 0.5) -> float:
    return sum(values) / len(values) if values else default


@dataclass
class FeatureState:
    team_form: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=5)))
    team_long_form: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=20)))
    venue_form: dict[tuple[str, str], deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=12)))
    h2h_form: dict[tuple[str, str], deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=10)))
    team_wins: Counter = field(default_factory=Counter)
    team_matches: Counter = field(default_factory=Counter)
    venue_wins: Counter = field(default_factory=Counter)
    venue_matches: Counter = field(default_factory=Counter)
    h2h_wins: Counter = field(default_factory=Counter)
    h2h_matches: Counter = field(default_factory=Counter)
    elo: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    elo_history: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque([1500.0], maxlen=5)))


def _pair(team_a: str, team_b: str) -> tuple[str, str]:
    return tuple(sorted((team_a, team_b)))


def _streak(wins: list[int]) -> int:
    streak = 0
    for w in reversed(wins):
        if w == 1:
            streak += 1
        else:
            break
    return streak


def _team_features(state: FeatureState, team: str, venue: str | None) -> dict[str, float]:
    form = list(state.team_form[team])
    venue_key = (team, venue)
    long_form = list(state.team_long_form[team])
    venue_form = list(state.venue_form[venue_key])
    elo_hist = list(state.elo_history[team])
    
    recent_rr = _rate([item["runs_per_ball"] for item in form], 1.3)
    recent_cr = _rate([item["conceded_per_ball"] for item in form], 1.3)
    
    return {
        "elo": state.elo[team],
        "recent_win_rate": _rate([item["won"] for item in form]),
        "overall_win_rate": _rate(long_form),
        "recent_run_rate": recent_rr,
        "recent_concede_rate": recent_cr,
        "venue_win_rate": _rate(venue_form),
        "matches_played": state.team_matches[team],
        "elo_momentum": elo_hist[-1] - elo_hist[0] if elo_hist else 0.0,
        "win_streak": _streak(long_form),
        "net_run_rate": recent_rr - recent_cr,
    }


def make_feature_row(state: FeatureState, team_a: str, team_b: str, venue: str | None) -> dict[str, float]:
    """Return features available before a match. Missing history uses neutral priors."""
    a, b = _team_features(state, team_a, venue), _team_features(state, team_b, venue)
    pair = _pair(team_a, team_b)
    h2h_results = list(state.h2h_form[pair])
    h2h_a = _rate([int(winner == team_a) for winner in h2h_results])
    
    # recent 3 h2h matches
    h2h_recent = h2h_results[-3:] if h2h_results else []
    h2h_recent_a = _rate([int(winner == team_a) for winner in h2h_recent])
    
    return {
        "elo_difference": a["elo"] - b["elo"],
        "recent_win_rate_difference": a["recent_win_rate"] - b["recent_win_rate"],
        "overall_win_rate_difference": a["overall_win_rate"] - b["overall_win_rate"],
        "recent_run_rate_difference": a["recent_run_rate"] - b["recent_run_rate"],
        "recent_concede_rate_difference": a["recent_concede_rate"] - b["recent_concede_rate"],
        "head_to_head_advantage": h2h_a - 0.5,
        "venue_win_rate_difference": a["venue_win_rate"] - b["venue_win_rate"],
        "team_a_matches_played": a["matches_played"],
        "team_b_matches_played": b["matches_played"],
        
        "h2h_recent_momentum": h2h_recent_a - 0.5,
        "venue_dominance_score_diff": (a["venue_win_rate"] * min(a["matches_played"], 12)) - (b["venue_win_rate"] * min(b["matches_played"], 12)),
        "elo_momentum_diff": a["elo_momentum"] - b["elo_momentum"],
        "win_streak_diff": a["win_streak"] - b["win_streak"],
        "net_run_rate_diff": a["net_run_rate"] - b["net_run_rate"],
        
        # Cross matchup: A's batting vs B's bowling
        "batting_vs_bowling_matchup_a": a["recent_run_rate"] - b["recent_concede_rate"],
        # Cross matchup: B's batting vs A's bowling
        "batting_vs_bowling_matchup_b": b["recent_run_rate"] - a["recent_concede_rate"],
    }


def _update(state: FeatureState, match: pd.Series, stats: dict[tuple[str, str], dict[str, float]]) -> None:
    team_a, team_b, venue, winner = match.team_a, match.team_b, match.venue, match.winner
    pair = _pair(team_a, team_b)
    for team, opponent in ((team_a, team_b), (team_b, team_a)):
        own, other = stats.get((match.match_id, team), {}), stats.get((match.match_id, opponent), {})
        state.team_matches[team] += 1
        won = int(team == winner)
        state.team_wins[team] += won
        state.venue_matches[(team, venue)] += 1
        state.venue_wins[(team, venue)] += won
        state.team_form[team].append({
            "won": won,
            "runs_per_ball": own.get("runs_scored", 0) / max(own.get("balls_faced", 0), 1),
            "conceded_per_ball": other.get("runs_scored", 0) / max(other.get("balls_faced", 0), 1),
        })
        state.team_long_form[team].append(won)
        state.venue_form[(team, venue)].append(won)
    state.h2h_matches[pair] += 1
    state.h2h_wins[(pair, winner)] += 1
    state.h2h_form[pair].append(winner)
    expected_a = 1 / (1 + 10 ** ((state.elo[team_b] - state.elo[team_a]) / 400))
    actual_a = int(winner == team_a)
    state.elo[team_a] += 20 * (actual_a - expected_a)
    state.elo[team_b] += 20 * ((1 - actual_a) - (1 - expected_a))
    state.elo_history[team_a].append(state.elo[team_a])
    state.elo_history[team_b].append(state.elo[team_b])


def build_feature_table(matches: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """Create one label-ready row per decisive IPL match without future-data leakage."""
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.dropna(subset=["winner", "date", "venue"]).sort_values(["date", "match_id"])
    stats = {(row.match_id, row.team): row._asdict() for row in team_stats.itertuples(index=False)}
    state, rows, active_season = FeatureState(), [], None
    # Matches on the same date receive features from the same prior state.
    for _, day_matches in matches.groupby("date", sort=True):
        season = day_matches.iloc[0].season
        if active_season is not None and season != active_season:
            # Regress ratings to neutral between seasons so old squads do not
            # dictate the following year's prediction.
            for team, rating in state.elo.items():
                new_rating = 1500 + 0.5 * (rating - 1500)
                state.elo[team] = new_rating
                state.elo_history[team].append(new_rating)
        active_season = season
        pending = []
        for match in day_matches.itertuples(index=False):
            row = {"match_id": match.match_id, "date": match.date, "season": match.season,
                   "team_a": match.team_a, "team_b": match.team_b, "venue": match.venue,
                   "target_team_a_win": int(match.winner == match.team_a)}
            row.update(make_feature_row(state, match.team_a, match.team_b, match.venue))
            pending.append((pd.Series(match._asdict()), row))
        rows.extend(row for _, row in pending)
        for match, _ in pending:
            _update(state, match, stats)
    return pd.DataFrame(rows)


def build_state_before(matches: pd.DataFrame, team_stats: pd.DataFrame, cutoff: pd.Timestamp) -> FeatureState:
    """Replay completed decisive matches strictly before *cutoff* for live inference."""
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.dropna(subset=["winner", "date", "venue"])
    matches = matches[matches["date"] < cutoff].sort_values(["date", "match_id"])
    stats = {(row.match_id, row.team): row._asdict() for row in team_stats.itertuples(index=False)}
    state, active_season = FeatureState(), None
    for _, day_matches in matches.groupby("date", sort=True):
        season = day_matches.iloc[0].season
        if active_season is not None and season != active_season:
            for team, rating in state.elo.items():
                new_rating = 1500 + 0.5 * (rating - 1500)
                state.elo[team] = new_rating
                state.elo_history[team].append(new_rating)
        active_season = season
        for match in day_matches.itertuples(index=False):
            _update(state, pd.Series(match._asdict()), stats)
    return state


def build_and_save() -> Path:
    matches = pd.read_csv(PROCESSED_DIR / "ipl_matches.csv")
    team_stats = pd.read_csv(PROCESSED_DIR / "ipl_team_match_stats.csv")
    features = build_feature_table(matches, team_stats)
    output = PROCESSED_DIR / "ipl_match_features.csv"
    features.to_csv(output, index=False)
    return output


if __name__ == "__main__":
    print(build_and_save())
