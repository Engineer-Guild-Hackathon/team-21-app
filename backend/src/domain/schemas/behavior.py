from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from ..models.behavior import ActionType

class BehaviorBase(BaseModel):
    """行動の基本スキーマ"""
    action_type: ActionType
    problem_id: Optional[int] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    attempt_count: int = 1
    success: Optional[bool] = None
    approach_description: Optional[str] = None
    emotion_state: Optional[str] = None

class BehaviorCreate(BehaviorBase):
    """行動作成スキーマ"""
    pass

class BehaviorResponse(BehaviorBase):
    """行動レスポンススキーマ"""
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class BehaviorStats(BaseModel):
    """行動統計スキーマ"""
    total_problems_attempted: int
    average_attempts_per_problem: float
    success_rate: float
    average_time_per_problem: float
    hint_usage_rate: float
    give_up_rate: float
    collaboration_count: int
    reflection_count: int
