from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.core.security import get_current_active_user
from src.domain.models.emotion import Emotion
from src.domain.models.user import User
from src.domain.schemas.emotion import (
    EmotionAnalysisRequest,
    EmotionAnalysisResponse,
    EmotionResponse,
)
from src.infrastructure.database import get_db

router = APIRouter()


@router.post("/analyze", response_model=EmotionAnalysisResponse)
async def analyze_emotion(
    request: EmotionAnalysisRequest,
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """感情分析を実行"""
    # MLサービスに分析リクエストを送信
    analysis_result = {
        "emotions": {
            "joy": 0.8,
            "sadness": 0.1,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.1,
            "frustration": 0.0,
            "concentration": 0.7,
        },
        "feedback": "とても良い進捗です！この調子で続けましょう。",
        "next_action": {
            "type": "challenge",
            "description": "より難しい問題に挑戦する準備ができているようです。",
        },
    }

    # 分析結果を保存
    emotion_record = Emotion(
        user_id=current_user.id,
        emotion_type=max(analysis_result["emotions"].items(), key=lambda x: x[1])[0],
        intensity=max(analysis_result["emotions"].values()),
        context=request.context,
    )
    db.add(emotion_record)
    db.commit()
    db.refresh(emotion_record)

    return analysis_result


@router.get("/history", response_model=List[EmotionResponse])
async def get_emotion_history(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> List[Emotion]:
    """感情履歴を取得"""
    query = db.query(Emotion).filter(Emotion.user_id == current_user.id)

    if start_date:
        query = query.filter(Emotion.created_at >= start_date)
    if end_date:
        query = query.filter(Emotion.created_at <= end_date)

    return query.all()


@router.get("/stats", response_model=Dict[str, Any])
async def get_emotion_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """感情統計を取得"""
    query = db.query(Emotion).filter(Emotion.user_id == current_user.id)

    if start_date:
        query = query.filter(Emotion.created_at >= start_date)
    if end_date:
        query = query.filter(Emotion.created_at <= end_date)

    # 感情タイプごとの集計
    emotion_counts: Dict[str, int] = {}
    records = query.all()
    for record in records:
        emotion_counts[record.emotion_type] = (
            emotion_counts.get(record.emotion_type, 0) + 1
        )

    # 平均強度の計算
    avg_intensity = query.with_entities(func.avg(Emotion.intensity)).scalar() or 0.0

    return {
        "emotion_counts": emotion_counts,
        "total_records": len(records),
        "average_intensity": float(avg_intensity),
    }
