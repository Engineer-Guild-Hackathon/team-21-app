import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1 import auth, emotion_analysis, emotions, feedback, learning, users

app = FastAPI(
    title="非認知能力学習プラットフォーム API",
    description="AIを活用した非認知能力の学習・トレーニングプラットフォームのAPI",
    version="1.0.0",
)


# CORS設定 - 環境変数から動的に読み取り
def get_allowed_origins():
    """環境変数から許可されたオリジンを取得"""
    # デフォルトのオリジン（開発環境用）
    default_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # 環境変数から追加のオリジンを取得
    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
    if allowed_origins_env:
        # カンマ区切りで複数のオリジンが設定されている場合
        additional_origins = [
            origin.strip() for origin in allowed_origins_env.split(",")
        ]
        return default_origins + additional_origins

    # 本番環境用のデフォルトオリジンも追加
    production_origins = [
        "http://app.34.107.156.246.nip.io",
        "https://app.34.107.156.246.nip.io",
    ]

    return default_origins + production_origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターの登録
app.include_router(auth.router, prefix="/api/auth", tags=["認証"])
app.include_router(users.router, prefix="/api/users", tags=["ユーザー"])
app.include_router(
    emotion_analysis.router, prefix="/api/emotion-analysis", tags=["感情分析"]
)
app.include_router(emotions.router, prefix="/api/emotions", tags=["感情"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["フィードバック"])
app.include_router(learning.router, prefix="/api/learning", tags=["学習"])


@app.get("/")
async def root() -> dict[str, str]:
    """ルートエンドポイント"""
    return {"message": "非認知能力学習プラットフォーム API"}
