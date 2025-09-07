from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import auth, users, learning, emotions, feedback

app = FastAPI(title="非認知能力学習プラットフォーム API")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # フロントエンドのURL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターの登録
app.include_router(auth.router, prefix="/api/auth", tags=["認証"])
app.include_router(users.router, prefix="/api/users", tags=["ユーザー"])
app.include_router(learning.router, prefix="/api/learning", tags=["学習"])
app.include_router(emotions.router, prefix="/api/emotions", tags=["感情分析"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["フィードバック"])

@app.get("/")
async def root():
    return {"message": "非認知能力学習プラットフォーム API"}