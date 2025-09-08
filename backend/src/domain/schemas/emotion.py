from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel

class EmotionBase(BaseModel):
    """感情の基本スキーマ"""
    emotion_type: str
    intensity: float
    context: Optional[str] = None

class EmotionCreate(EmotionBase):
    """感情作成スキーマ"""
    pass

class EmotionResponse(EmotionBase):
    """感情レスポンススキーマ"""
    id: int
    user_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class EmotionAnalysisRequest(BaseModel):
    """感情分析リクエストスキーマ"""
    text: Optional[str] = None
    context: Optional[str] = None
    behavior_data: Optional[Dict[str, float]] = None

class EmotionAnalysisResponse(BaseModel):
    """感情分析レスポンススキーマ"""
    emotions: Dict[str, float]
    feedback: str
    next_action: Dict[str, str]
