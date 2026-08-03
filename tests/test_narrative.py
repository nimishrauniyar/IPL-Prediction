from src.rag.narrative import generate_grounded_preview


def test_preview_is_grounded_in_prediction_data(monkeypatch):
    monkeypatch.setattr("src.rag.narrative.player_documents", lambda *args: [{"id": "one", "text": "A player scored 50 runs."}])
    preview = generate_grounded_preview({"team_a": "A", "team_b": "B", "team_a_win_probability": 0.6,
        "team_b_win_probability": 0.4, "confidence": "medium", "mode": "hypothetical_match", "match_date": None})
    assert "A player scored 50 runs." in preview["narrative"]
