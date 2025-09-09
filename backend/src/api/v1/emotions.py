from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from srccore.security import get_current_active_user
from srcdomain.models.user import User
from srcdomain.schemas.emotion import (
    EmotionCreate,
    EmotionResponse,
    EmotionTrendResponse,
)
from srcdomain.types.emotion import EmotionAnalysis, UserId
from srcinfrastructure.database import get_db
from srcinfrastructure.repositories.emotion_repository import (
    SQLAlchemyEmotionRepository,
)
from srcservices.emotion.emotion_service import EmotionService

router = APIRouter()


def get_emotion_service(db: Session = Depends(get_db)) -> EmotionService:
    """EmotionServiceの依存性注入"""
    repository = SQLAlchemyEmotionRepository(db)
    return EmotionService(repository)


@router.post("/analyze", response_model=EmotionResponse)
async def analyze_emotion(
    data: EmotionCreate,
    service: EmotionService = Depends(get_emotion_service),
    current_user: User = Depends(get_current_active_user),
):
    """感情分析を実行し結果を保存"""
    analysis = EmotionAnalysis(
        scores=data.scores,
        dominant_emotion=data.dominant_emotion,
        timestamp=data.timestamp,
        source_type=data.source_type,
        source_content=data.content,
    )

    emotion = await service.record_emotion(UserId(current_user.id), analysis)
    return emotion


@router.get("/me/latest", response_model=EmotionResponse)
async def get_my_latest_emotion(
    service: EmotionService = Depends(get_emotion_service),
    current_user: User = Depends(get_current_active_user),
):
    """現在のユーザーの最新の感情状態を取得"""
    emotion = await service.get_latest_emotion(UserId(current_user.id))
    if not emotion:
        raise HTTPException(status_code=404, detail="No emotion data found")
    return emotion


@router.get("/me/history", response_model=list[EmotionResponse])
async def get_my_emotion_history(
    days: int = 7,
    service: EmotionService = Depends(get_emotion_service),
    current_user: User = Depends(get_current_active_user),
):
    """現在のユーザーの感情履歴を取得"""
    return await service.get_user_emotions(UserId(current_user.id), days=days)


@router.get("/me/trend", response_model=EmotionTrendResponse)
async def get_my_emotion_trend(
    period_days: int = 7,
    service: EmotionService = Depends(get_emotion_service),
    current_user: User = Depends(get_current_active_user),
):
    """現在のユーザーの感情トレンドを分析"""
    trend = await service.analyze_trend(
        UserId(current_user.id), period_days=period_days
    )
    if not trend:
        raise HTTPException(
            status_code=404, detail="Not enough data for trend analysis"
        )
    return trend


@router.get("/similar/{emotion}", response_model=list[EmotionResponse])
async def find_similar_emotions(
    emotion: str,
    threshold: float = 0.5,
    service: EmotionService = Depends(get_emotion_service),
    current_user: User = Depends(get_current_active_user),
):
    """特定の感情に類似したデータを検索"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return await service.find_similar_emotions(emotion, threshold=threshold)
