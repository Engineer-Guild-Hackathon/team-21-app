from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.v1 import learning

app = FastAPI()
app.include_router(learning.router, prefix="/api/learning")
client = TestClient(app)


def test_post_learn_action_and_get_summary():
    # 1) イベント送信
    payload = {
        "event_id": "e1",
        "user_id": "user_001",
        "session_id": "s1",
        "action": "retry",
        "think_time_ms": 4200,
        "success": True,
        "difficulty": "normal",
    }

    res = client.post("/api/learning/events/learn-action", json=payload)
    assert res.status_code == 201
    assert res.json()["status"] == "accepted"

    # 2) サマリ取得
    res2 = client.get(
        "/api/learning/metrics/noncog-summary", params={"user_id": "user_001"}
    )
    assert res2.status_code == 200
    data = res2.json()
    assert data["user_id"] == "user_001"
    assert data["retry_count"] >= 1
    assert data["avg_think_time_ms"] > 0


def test_validation_error_on_invalid_action():
    payload = {
        "event_id": "e2",
        "user_id": "user_001",
        "session_id": "s1",
        "action": "invalid_action",
        "think_time_ms": 1000,
    }
    res = client.post("/api/learning/events/learn-action", json=payload)
    # Pydantic のバリデーションにより 422
    assert res.status_code == 422
