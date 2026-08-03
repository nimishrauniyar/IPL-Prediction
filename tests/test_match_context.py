from datetime import date

import pandas as pd

from src.context.match_context import get_match_context


def test_resolves_next_fixture_with_aliases(tmp_path):
    schedule = tmp_path / "schedule.csv"
    pd.DataFrame([{"date": "2026-04-05", "team_a": "Chennai Super Kings", "team_b": "Mumbai Indians", "venue": "Chepauk"}]).to_csv(schedule, index=False)
    context = get_match_context("CSK", "MI", schedule, date(2026, 4, 1))
    assert context.mode == "scheduled_match"
    assert context.venue == "Chepauk"


def test_returns_neutral_context_when_fixture_is_missing(tmp_path):
    schedule = tmp_path / "schedule.csv"
    pd.DataFrame(columns=["date", "team_a", "team_b", "venue"]).to_csv(schedule, index=False)
    context = get_match_context("CSK", "MI", schedule, date(2026, 4, 1))
    assert context.mode == "hypothetical_match"
    assert context.venue == "Neutral venue"
