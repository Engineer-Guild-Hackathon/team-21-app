from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..core.security import get_current_active_user
from ..domain.models.behavior import ActionType, BehaviorRecord
from ..domain.models.user import User
from ..domain.schemas.behavior import BehaviorCreate, BehaviorResponse, BehaviorStats
from ..infrastructure.database import get_db

router = APIRouter()


@router.post("/record", response_model=BehaviorResponse)
async def record_behavior(
    behavior: BehaviorCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> BehaviorRecord:
    """行動を記録"""
    db_behavior = BehaviorRecord(
        user_id=current_user.id,
        problem_id=behavior.problem_id,
        action_type=behavior.action_type,
        start_time=behavior.start_time,
        end_time=behavior.end_time,
        attempt_count=behavior.attempt_count,
        success=behavior.success,
        approach_description=behavior.approach_description,
        emotion_state=behavior.emotion_state,
    )
    db.add(db_behavior)
    db.commit()
    db.refresh(db_behavior)
    return db_behavior


@router.get("/stats", response_model=BehaviorStats)
async def get_behavior_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> dict:
    """行動統計を取得"""
    query = db.query(BehaviorRecord).filter(BehaviorRecord.user_id == current_user.id)

    if start_date:
        query = query.filter(BehaviorRecord.created_at >= start_date)
    if end_date:
        query = query.filter(BehaviorRecord.created_at <= end_date)

    # 基本統計の計算
    total_problems = query.filter(
        BehaviorRecord.action_type == ActionType.PROBLEM_START
    ).count()
    total_attempts = query.filter(
        BehaviorRecord.action_type == ActionType.PROBLEM_SUBMIT
    ).count()
    successful_attempts = query.filter(
        BehaviorRecord.action_type == ActionType.PROBLEM_SUBMIT,
        BehaviorRecord.success == True,
    ).count()

    # 時間の計算
    time_records = query.filter(
        BehaviorRecord.action_type == ActionType.PROBLEM_SUBMIT,
        BehaviorRecord.end_time.isnot(None),
    ).all()
    total_time = sum(
        (record.end_time - record.start_time).total_seconds()
        for record in time_records
        if record.end_time and record.start_time
    )

    # その他の統計
    hint_requests = query.filter(
        BehaviorRecord.action_type == ActionType.HINT_REQUEST
    ).count()
    give_ups = query.filter(BehaviorRecord.action_type == ActionType.GIVE_UP).count()
    collaborations = query.filter(
        BehaviorRecord.action_type == ActionType.COLLABORATION
    ).count()
    reflections = query.filter(
        BehaviorRecord.action_type == ActionType.REFLECTION
    ).count()

    return {
        "total_problems_attempted": total_problems,
        "average_attempts_per_problem": (
            total_attempts / total_problems if total_problems > 0 else 0
        ),
        "success_rate": (
            successful_attempts / total_attempts if total_attempts > 0 else 0
        ),
        "average_time_per_problem": (
            total_time / total_problems if total_problems > 0 else 0
        ),
        "hint_usage_rate": hint_requests / total_problems if total_problems > 0 else 0,
        "give_up_rate": give_ups / total_problems if total_problems > 0 else 0,
        "collaboration_count": collaborations,
        "reflection_count": reflections,
    }


@router.get("/history", response_model=List[BehaviorResponse])
async def get_behavior_history(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    action_type: Optional[ActionType] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> List[BehaviorRecord]:
    """行動履歴を取得"""
    query = db.query(BehaviorRecord).filter(BehaviorRecord.user_id == current_user.id)

    if start_date:
        query = query.filter(BehaviorRecord.created_at >= start_date)
    if end_date:
        query = query.filter(BehaviorRecord.created_at <= end_date)
    if action_type:
        query = query.filter(BehaviorRecord.action_type == action_type)

    return query.order_by(BehaviorRecord.created_at.desc()).all()
