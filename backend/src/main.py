from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1 import auth, emotion_analysis, emotions, feedback, learning, users

app = FastAPI(
    title="非認知能力学習プラットフォーム API",
    description="AIを活用した非認知能力の学習・トレーニングプラットフォームのAPI",
    version="1.0.0",
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # フロントエンドのURLを明示的に指定
        "http://localhost:8000",  # 開発環境のバックエンドURL
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-CSRF-Token"],
    expose_headers=["Content-Type", "Authorization"],
)

# ルーターの登録
app.include_router(auth.router, prefix="/api/v1", tags=["認証"])
app.include_router(users.router, prefix="/api/v1/users", tags=["ユーザー"])
app.include_router(
    emotion_analysis.router, prefix="/api/v1/emotion-analysis", tags=["感情分析"]
)
app.include_router(emotions.router, prefix="/api/v1/emotions", tags=["感情"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["フィードバック"])
app.include_router(learning.router, prefix="/api/v1/learning", tags=["学習"])


@app.get("/")
async def root() -> dict[str, str]:
    """ルートエンドポイント"""
    return {"message": "非認知能力学習プラットフォーム API"}
