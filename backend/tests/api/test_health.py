"""
Tests for health endpoint
"""

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """
    Test that the health endpoint returns 200 OK.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
