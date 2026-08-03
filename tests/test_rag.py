from src.rag.retriever import retrieve


def test_retrieval_returns_ranked_evidence():
    documents = [
        {"id": "a", "text": "Chennai player has strong batting form and high strike rate."},
        {"id": "b", "text": "Mumbai bowler has low economy and recent wickets."},
    ]
    results = retrieve("Mumbai bowling wickets economy", documents, limit=1)
    assert results[0]["id"] == "b"
