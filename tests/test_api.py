"""
Integration tests for NutriScore Flask REST API & Web Application
"""

import pytest
import json
from app_server import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_healthcheck_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert "service" in data
    assert "uptime_seconds" in data
    assert "model_version" in data


def test_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "total_requests" in data
    assert "error_count" in data
    assert "latency_ms" in data
    assert "grade_distribution" in data


def test_predict_endpoint_valid(client):
    payload = {
        "calories": 200,
        "protein_g": 18.0,
        "carbs_g": 12.0,
        "fiber_g": 4.0,
        "fat_g": 5.0,
        "sugar_g": 2.0,
        "sodium_mg": 150.0,
        "ingredients": "Greek Yogurt, Honey, Blueberries"
    }
    response = client.post(
        "/api/v1/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert "score" in data
    assert "grade" in data
    assert data["grade"] in ["A", "B", "C", "D", "E"]


def test_predict_endpoint_invalid_content_type(client):
    response = client.post("/api/v1/predict", data="not json")
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data["success"] is False


def test_serve_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"NutriScore" in response.data
