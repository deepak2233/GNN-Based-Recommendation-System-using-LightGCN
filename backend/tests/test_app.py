from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_recommend():
    response = client.post("/recommend/", json={"user_id": 0, "top_k": 10})
    assert response.status_code == 200
    assert "recommended_items" in response.json()
