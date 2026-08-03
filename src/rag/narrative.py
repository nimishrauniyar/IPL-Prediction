"""Generate a grounded local match narrative from retrieved evidence."""

from __future__ import annotations

from src.rag.retriever import player_documents, retrieve


def generate_grounded_preview(prediction: dict) -> dict:
    team_a, team_b = prediction["team_a"], prediction["team_b"]
    query = f"{team_a} {team_b} IPL current form batting bowling key players"
    evidence = retrieve(query, player_documents(team_a, team_b, prediction.get("match_date")), limit=6)
    favored = team_a if prediction["team_a_win_probability"] >= 0.5 else team_b
    probability = max(prediction["team_a_win_probability"], prediction["team_b_win_probability"])
    player_lines = [item["text"] for item in evidence[:3]]
    narrative = (f"{favored} are the model's slight pre-match favourites at {probability:.1%} probability "
                 f"({prediction['confidence']} confidence). This is a {prediction['mode'].replace('_', ' ')} prediction. "
                 f"Key retrieved IPL evidence: {' '.join(player_lines)}")
    return {"narrative": narrative, "evidence": evidence,
            "disclaimer": "The narrative is grounded only in the retrieved IPL records; inferred players are not confirmed playing XIs."}
