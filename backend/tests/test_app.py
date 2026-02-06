"""
Tests for the FastAPI application endpoints.

Covers health checks, single/batch recommendations,
input validation, and error handling.
"""

import pytest
from unittest.mock import MagicMock
import torch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_app_state():
    """Create a mock app state with model loaded."""
    from backend.app import app_state

    # Create mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = torch.tensor([5, 3, 7, 1, 9])
    mock_model.num_users = 100
    mock_model.num_items = 50

    # Mock forward pass for batch
    mock_model.return_value = (
        torch.randn(100, 32),  # user embeddings
        torch.randn(50, 32),   # item embeddings
    )

    # Save original state
    original_state = {
        'model': app_state.model,
        'edge_index': app_state.edge_index,
        'num_users': app_state.num_users,
        'num_items': app_state.num_items,
        'is_ready': app_state.is_ready,
        'device': app_state.device,
    }

    # Set mock state
    app_state.model = mock_model
    app_state.edge_index = torch.tensor([[0, 1], [1, 0]])
    app_state.num_users = 100
    app_state.num_items = 50
    app_state.is_ready = True
    app_state.device = torch.device('cpu')
    app_state.model_load_time = 1.5

    yield app_state

    # Restore original state
    for key, value in original_state.items():
        setattr(app_state, key, value)


@pytest.fixture
def client(mock_app_state):
    """Create a test client with mocked state."""
    from backend.app import app
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_ready(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["num_users"] == 100
        assert data["num_items"] == 50

    def test_health_check_not_ready(self, client, mock_app_state):
        mock_app_state.is_ready = False
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


class TestRecommendEndpoint:
    """Tests for the /recommend/ endpoint."""

    def test_recommend_success(self, client):
        response = client.post("/recommend/", json={"user_id": 0, "top_k": 5})
        assert response.status_code == 200
        data = response.json()
        assert "recommended_items" in data
        assert "user_id" in data
        assert data["user_id"] == 0
        assert "num_recommendations" in data

    def test_recommend_default_top_k(self, client):
        response = client.post("/recommend/", json={"user_id": 0})
        assert response.status_code == 200

    def test_recommend_invalid_user_id(self, client):
        response = client.post("/recommend/", json={"user_id": 999, "top_k": 5})
        assert response.status_code == 400
        assert "Invalid user_id" in response.json()["detail"]

    def test_recommend_negative_user_id(self, client):
        response = client.post("/recommend/", json={"user_id": -1, "top_k": 5})
        assert response.status_code == 422  # Pydantic validation

    def test_recommend_invalid_top_k_zero(self, client):
        response = client.post("/recommend/", json={"user_id": 0, "top_k": 0})
        assert response.status_code == 422

    def test_recommend_invalid_top_k_too_large(self, client):
        response = client.post("/recommend/", json={"user_id": 0, "top_k": 200})
        assert response.status_code == 422

    def test_recommend_model_not_loaded(self, client, mock_app_state):
        mock_app_state.is_ready = False
        response = client.post("/recommend/", json={"user_id": 0, "top_k": 5})
        assert response.status_code == 503


class TestBatchRecommendEndpoint:
    """Tests for the /recommend/batch endpoint."""

    def test_batch_recommend_success(self, client):
        response = client.post(
            "/recommend/batch",
            json={"user_ids": [0, 1, 2], "top_k": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) == 3

    def test_batch_recommend_invalid_user(self, client):
        response = client.post(
            "/recommend/batch",
            json={"user_ids": [0, 999], "top_k": 5}
        )
        assert response.status_code == 400

    def test_batch_recommend_empty_list(self, client):
        response = client.post(
            "/recommend/batch",
            json={"user_ids": [], "top_k": 5}
        )
        assert response.status_code == 422

    def test_batch_recommend_model_not_loaded(self, client, mock_app_state):
        mock_app_state.is_ready = False
        response = client.post(
            "/recommend/batch",
            json={"user_ids": [0], "top_k": 5}
        )
        assert response.status_code == 503
