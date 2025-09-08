from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..core.security import get_current_active_user
from ..domain.models.user import User
from ..domain.models.emotion import EmotionRecord
from ..domain.schemas.emotion import (
    EmotionCreate,
    EmotionResponse,
    EmotionStats
)
from ..infrastructure.database import get_db

router = APIRouter()

@router.post("/emotions/", response_model=EmotionResponse)
async def record_emotion(
    emotion: EmotionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> EmotionRecord:
    """感情を記録"""
    db_emotion = EmotionRecord(
        user_id=current_user.id,
        emotion_type=emotion.emotion_type,
        intensity=emotion.intensity,
        context=emotion.context
    )
    db.add(db_emotion)
    db.commit()
    db.refresh(db_emotion)
    return db_emotion

@router.get("/emotions/stats", response_model=EmotionStats)
async def get_emotion_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """感情統計を取得"""
    query = db.query(EmotionRecord).filter(EmotionRecord.user_id == current_user.id)
    
    if start_date:
        query = query.filter(EmotionRecord.created_at >= start_date)
    if end_date:
        query = query.filter(EmotionRecord.created_at <= end_date)
    
    # 感情タイプごとの集計
    emotion_counts: Dict[str, int] = {}
    records = query.all()
    for record in records:
        emotion_counts[record.emotion_type] = emotion_counts.get(record.emotion_type, 0) + 1
    
    # 平均強度の計算
    avg_intensity = query.with_entities(func.avg(EmotionRecord.intensity)).scalar() or 0.0
    
    return {
        "emotion_counts": emotion_counts,
        "total_records": len(records),
        "average_intensity": float(avg_intensity)
    }