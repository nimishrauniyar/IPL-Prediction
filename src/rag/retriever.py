"""Create evidence documents and retrieve them with local TF-IDF similarity."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.player_intelligence.player_impact import team_player_impacts


def player_documents(team_a: str, team_b: str, cutoff=None) -> list[dict]:
    documents = []
    for team, players in team_player_impacts(team_a, team_b, cutoff).items():
        for player in players:
            text = (f"{team} player {player['player']}: last {player['matches']} IPL appearances; "
                    f"{int(player['runs'])} runs at strike rate {player['strike_rate']:.1f}; "
                    f"{int(player['wickets'])} wickets at economy {player['economy']:.2f}; impact score {player['impact_score']}.")
            documents.append({"id": f"{team}:{player['player']}", "team": team, "text": text, "stats": player})
    return documents


def retrieve(query: str, documents: list[dict], limit: int = 6) -> list[dict]:
    if not documents:
        return []
    corpus = [doc["text"] for doc in documents]
    matrix = TfidfVectorizer(stop_words="english").fit_transform(corpus + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = scores.argsort()[::-1][:limit]
    return [{**documents[index], "relevance": round(float(scores[index]), 4)} for index in ranked]
