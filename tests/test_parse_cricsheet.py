import json
import zipfile

from src.ingestion.parse_cricsheet import parse_archive


def test_parses_minimal_cricsheet_match(tmp_path):
    payload = {
        "info": {"dates": ["2025-04-01"], "teams": ["A", "B"], "venue": "Ground", "outcome": {"winner": "A"}},
        "innings": [{"team": "A", "overs": [{"over": 0, "deliveries": [{
            "batter": "Batter", "bowler": "Bowler", "runs": {"batter": 4, "extras": 0, "total": 4}
        }]}]}],
    }
    archive = tmp_path / "ipl.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("123.json", json.dumps(payload))
    tables = parse_archive(archive)
    assert tables["matches"].iloc[0].winner == "A"
    assert tables["deliveries"].iloc[0].batter_runs == 4
    assert tables["team_match_stats"].query("team == 'A'").iloc[0].runs_scored == 4
