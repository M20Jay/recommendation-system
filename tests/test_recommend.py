"""
Week 7 — Recommendation System
API Tests

Author: Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya
GitHub: github.com/M20Jay
Date: May 2026
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test /health returns 200 and correct fields."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "version" in data
    assert "timestamp" in data


def test_recommend_valid_user():
    """Test /recommend returns recommendations for valid user."""
    response = client.post(
        "/recommend",
        json={"user_id": 196, "n": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 196
    assert len(data["recommendations"]) == 5
    assert "model" in data
    assert "timestamp" in data


def test_recommend_returns_correct_fields():
    """Test each recommendation has required fields."""
    response = client.post(
        "/recommend",
        json={"user_id": 196, "n": 3}
    )
    assert response.status_code == 200
    recs = response.json()["recommendations"]
    for rec in recs:
        assert "movie_id" in rec
        assert "title" in rec
        assert "predicted_rating" in rec
        assert 1.0 <= rec["predicted_rating"] <= 5.0


def test_recommend_invalid_user():
    """Test /recommend returns 404 for unknown user."""
    response = client.post(
        "/recommend",
        json={"user_id": 99999, "n": 5}
    )
    assert response.status_code == 404


def test_recommend_default_n():
    """Test /recommend returns 10 by default."""
    response = client.post(
        "/recommend",
        json={"user_id": 196}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 10


def test_recommend_different_users():
    """Test recommendations differ between users."""
    response_196 = client.post(
        "/recommend",
        json={"user_id": 196, "n": 5}
    )
    response_405 = client.post(
        "/recommend",
        json={"user_id": 405, "n": 5}
    )
    recs_196 = [r["movie_id"] for r in
                response_196.json()["recommendations"]]
    recs_405 = [r["movie_id"] for r in
                response_405.json()["recommendations"]]

    assert recs_196 != recs_405