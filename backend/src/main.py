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
    allow_origins=["*"],  # 本番環境では適切なオリジンを指定
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
