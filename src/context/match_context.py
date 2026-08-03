"""Resolve an upcoming IPL fixture from a verified local schedule."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd

from src.config import EXTERNAL_DIR

TEAM_ALIASES = {
    "CSK": "Chennai Super Kings", "MI": "Mumbai Indians", "RCB": "Royal Challengers Bengaluru",
    "KKR": "Kolkata Knight Riders", "RR": "Rajasthan Royals", "DC": "Delhi Capitals",
    "PBKS": "Punjab Kings", "SRH": "Sunrisers Hyderabad", "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants",
}


@dataclass(frozen=True)
class MatchContext:
    team_a: str
    team_b: str
    venue: str
    match_date: str | None
    mode: str
    message: str


def canonical_team(name: str) -> str:
    cleaned = " ".join(name.strip().split())
    return TEAM_ALIASES.get(cleaned.upper(), cleaned)


def get_match_context(team_a: str, team_b: str, schedule_path=None, today: date | None = None) -> MatchContext:
    team_a, team_b = canonical_team(team_a), canonical_team(team_b)
    if team_a == team_b:
        raise ValueError("Choose two different IPL teams.")
    schedule_path = schedule_path or EXTERNAL_DIR / "ipl_schedule.csv"
    schedule = pd.read_csv(schedule_path)
    required = {"date", "team_a", "team_b", "venue"}
    if not required.issubset(schedule.columns):
        raise ValueError(f"Schedule must contain {sorted(required)}")
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
    today = today or date.today()
    mask = (((schedule.team_a == team_a) & (schedule.team_b == team_b)) |
            ((schedule.team_a == team_b) & (schedule.team_b == team_a)))
    fixtures = schedule.loc[mask & (schedule["date"].dt.date >= today)].sort_values("date")
    if fixtures.empty:
        return MatchContext(team_a, team_b, "Neutral venue", None, "hypothetical_match",
                            "No future fixture was found in the local schedule; using a neutral venue.")
    fixture = fixtures.iloc[0]
    return MatchContext(team_a, team_b, fixture.venue, fixture.date.date().isoformat(), "scheduled_match",
                        "Venue and date were resolved from the local IPL schedule.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a team-only IPL matchup.")
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    args = parser.parse_args()
    print(asdict(get_match_context(args.team_a, args.team_b)))


if __name__ == "__main__":
    main()
