"""Normalize Cricsheet JSON archives into match, delivery, team, and player tables."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR


def _runs(delivery: dict[str, Any], kind: str) -> int:
    return int(delivery.get("runs", {}).get(kind, 0))


def _iter_json(archive: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    with zipfile.ZipFile(archive) as zipped:
        for filename in zipped.namelist():
            if filename.endswith(".json") and not filename.endswith("README.json"):
                with zipped.open(filename) as file:
                    yield Path(filename).stem, json.load(file)


def parse_archive(archive: Path) -> dict[str, pd.DataFrame]:
    matches: list[dict[str, Any]] = []
    deliveries: list[dict[str, Any]] = []
    team_stats: list[dict[str, Any]] = []
    player_stats: list[dict[str, Any]] = []

    for match_id, payload in _iter_json(archive):
        info = payload["info"]
        teams = info.get("teams", [])
        outcome = info.get("outcome", {})
        match_date = info.get("dates", [None])[0]
        venue = info.get("venue")
        matches.append({
            "match_id": match_id,
            "date": match_date,
            "team_a": teams[0] if len(teams) > 0 else None,
            "team_b": teams[1] if len(teams) > 1 else None,
            "venue": venue,
            "city": info.get("city"),
            "season": info.get("season"),
            "stage": info.get("event", {}).get("stage") or "League stage",
            "winner": outcome.get("winner"),
            "result": outcome.get("result"),
            "toss_winner": info.get("toss", {}).get("winner"),
            "toss_decision": info.get("toss", {}).get("decision"),
        })

        team_totals: dict[str, Counter[str]] = defaultdict(Counter)
        player_totals: dict[str, Counter[str]] = defaultdict(Counter)
        for innings_index, innings in enumerate(payload.get("innings", []), start=1):
            batting_team = innings.get("team")
            for over in innings.get("overs", []):
                for delivery in over.get("deliveries", []):
                    batter = delivery["batter"]
                    bowler = delivery["bowler"]
                    runs_batter = _runs(delivery, "batter")
                    runs_total = _runs(delivery, "total")
                    extras = _runs(delivery, "extras")
                    team_totals[batting_team]["runs_scored"] += runs_total
                    team_totals[batting_team]["balls_faced"] += 1
                    team_totals[bowler]["runs_conceded"] += 0  # avoids missing team keys
                    player_totals[batter]["runs"] += runs_batter
                    player_totals[batter]["balls_faced"] += 1
                    player_totals[bowler]["balls_bowled"] += 1
                    player_totals[bowler]["runs_conceded"] += runs_total - extras
                    for wicket in delivery.get("wickets", []):
                        if wicket.get("kind") not in {"run out", "retired hurt", "obstructing the field"}:
                            player_totals[bowler]["wickets"] += 1
                    deliveries.append({
                        "match_id": match_id, "innings": innings_index, "batting_team": batting_team,
                        "over": over.get("over"), "batter": batter, "bowler": bowler,
                        "batter_runs": runs_batter, "extras": extras, "total_runs": runs_total,
                        "is_wicket": bool(delivery.get("wickets")),
                    })
        for team in teams:
            stats = team_totals[team]
            team_stats.append({"match_id": match_id, "team": team, "runs_scored": stats["runs_scored"],
                               "balls_faced": stats["balls_faced"], "won": team == outcome.get("winner")})
        player_teams = {player: team for team, players in info.get("players", {}).items() for player in players}
        for player, stats in player_totals.items():
            player_stats.append({"match_id": match_id, "player": player, "team": player_teams.get(player), **stats})

    return {"matches": pd.DataFrame(matches), "deliveries": pd.DataFrame(deliveries),
            "team_match_stats": pd.DataFrame(team_stats), "player_match_stats": pd.DataFrame(player_stats)}


def write_tables(dataset: str, archive: Path) -> None:
    for table_name, table in parse_archive(archive).items():
        table.to_csv(PROCESSED_DIR / f"{dataset}_{table_name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a downloaded Cricsheet archive.")
    parser.add_argument("--dataset", choices=("ipl", "t20i"), required=True)
    args = parser.parse_args()
    archive = RAW_DIR / f"{args.dataset}_json.zip"
    if not archive.exists():
        raise FileNotFoundError(f"Missing {archive}. Run download_cricsheet first.")
    write_tables(args.dataset, archive)


if __name__ == "__main__":
    main()
