from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint():
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_teams_endpoint_returns_known_team():
    response = TestClient(app).get("/api/teams")
    assert response.status_code == 200
    assert "Chennai Super Kings" in response.json()["teams"]
