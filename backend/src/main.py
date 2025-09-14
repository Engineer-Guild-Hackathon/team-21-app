import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1 import (
    auth,
    avatars,
    classes,
    emotion_analysis,
    emotions,
    feedback,
    learning,
    ml_integration,
    quests,
    users,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    # 起動時
    logging.info("app_startup_begin")

    # Kafkaコンシューマを開始
    try:
        from src.api.v1.learning import get_kafka_consumer

        consumer = get_kafka_consumer()
        if consumer.enabled:
            # バックグラウンドタスクでコンシューマを開始
            asyncio.create_task(consumer.start_consuming())
            logging.info("kafka_consumer_startup_task_created")
        else:
            logging.info("kafka_consumer_disabled_at_startup")
    except Exception as e:
        logging.exception("kafka_consumer_startup_error: %s", e)

    logging.info("app_startup_complete")

    yield

    # 終了時
    logging.info("app_shutdown_begin")

    # Kafkaコンシューマを停止
    try:
        from src.api.v1.learning import get_kafka_consumer

        consumer = get_kafka_consumer()
        consumer.stop_consuming()
        logging.info("kafka_consumer_shutdown_complete")
    except Exception as e:
        logging.exception("kafka_consumer_shutdown_error: %s", e)

    logging.info("app_shutdown_complete")


app = FastAPI(
    title="非認知能力学習プラットフォーム API",
    description="AIを活用した非認知能力の学習・トレーニングプラットフォームのAPI",
    version="1.0.0",
    lifespan=lifespan,
)


# CORS設定 - 環境変数から動的に読み取り
def get_allowed_origins() -> list[str]:
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

    # 環境変数が設定されていない場合、環境に応じてオリジンを決定
    environment = os.getenv("ENVIRONMENT", "development")

    if environment == "production":
        # 本番環境用のオリジン
        production_origins = [
            "http://app.34.107.156.246.nip.io",
            "https://app.34.107.156.246.nip.io",
        ]
        return default_origins + production_origins
    else:
        # 開発環境ではローカルオリジンのみ
        return default_origins


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
app.include_router(classes.router, prefix="/api/classes", tags=["クラス"])
app.include_router(
    emotion_analysis.router, prefix="/api/emotion-analysis", tags=["感情分析"]
)
app.include_router(emotions.router, prefix="/api/emotions", tags=["感情"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["フィードバック"])
app.include_router(learning.router, prefix="/api/learning", tags=["学習"])
app.include_router(quests.router, prefix="/api/quests", tags=["クエスト"])
app.include_router(avatars.router, prefix="/api/avatars", tags=["アバター・称号"])
app.include_router(ml_integration.router, prefix="/api/ml", tags=["ML統合"])


@app.get("/")
async def root() -> dict[str, str]:
    """ルートエンドポイント"""
    return {"message": "非認知能力学習プラットフォーム API"}
