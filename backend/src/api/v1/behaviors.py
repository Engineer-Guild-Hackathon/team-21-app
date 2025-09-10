from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from srcinfrastructure.database import get_db
from srcinfrastructure.repositories.behavior_repository import (
    SQLAlchemyBehaviorRepository,
)
from srcservices.behavior.behavior_service import BehaviorService

from src.core.security import get_current_active_user
from src.domain.models.user import User
from src.domain.schemas.behavior import (
    BehaviorCreate,
    BehaviorGoalCreate,
    BehaviorGoalResponse,
    BehaviorPatternResponse,
    BehaviorResponse,
    BehaviorStatisticsResponse,
    BehaviorTriggerResponse,
)
from src.domain.types.behavior import (
    BehaviorCategory,
    BehaviorGoal,
    BehaviorId,
    BehaviorMetrics,
    UserId,
)

router = APIRouter()


def get_behavior_service(db: Session = Depends(get_db)) -> BehaviorService:
    """BehaviorServiceの依存性注入"""
    repository = SQLAlchemyBehaviorRepository(db)
    return BehaviorService(repository)


@router.post("/record", response_model=BehaviorResponse)
async def record_behavior(
    data: BehaviorCreate,
    service: BehaviorService = Depends(get_behavior_service),
    current_user: User = Depends(get_current_active_user),
):
    """行動データを記録"""
    category = BehaviorCategory(
        name=data.category,
        description=data.category_description,
        impact_level=data.impact_level,
    )

    metrics = [
        BehaviorMetrics(
            metric_name=m.name,
            value=m.value,
            unit=m.unit,
            timestamp=m.timestamp,
        )
        for m in data.metrics
    ]

    behavior = await service.record_behavior(
        UserId(current_user.id),
        category,
        metrics,
        data.duration,
        data.notes,
    )
    return behavior


@router.get("/me/history", response_model=list[BehaviorResponse])
async def get_my_behavior_history(
    category: str = None,
    days: int = 7,
    service: BehaviorService = Depends(get_behavior_service),
    current_user: User = Depends(get_current_active_user),
):
    """現在のユーザーの行動履歴を取得"""
    category_obj = None
    if category:
        category_obj = BehaviorCategory(
            name=category,
            description="",
            impact_level=3,
        )

    return await service.get_user_behaviors(
        UserId(current_user.id),
        category=category_obj,
        days=days,
    )


@router.get("/me/patterns", response_model=list[BehaviorPatternResponse])
async def get_my_behavior_patterns(
    period_days: int = 30,
    service: BehaviorService = Depends(get_behavior_service),
    current_user: User = Depends(get_current_active_user),
):
    """現在のユーザーの行動パターンを分析"""
    return await service.analyze_patterns(
        UserId(current_user.id),
        period_days=period_days,
    )


@router.get("/me/statistics", response_model=BehaviorStatisticsResponse)
async def get_my_behavior_statistics(
    category: str = None,
    period_days: int = 7,
    service: BehaviorService = Depends(get_behavior_service),
    current_user: User = Depends(get_current_active_user),
):
    """現在のユーザーの行動統計を取得"""
    category_obj = None
    if category:
        category_obj = BehaviorCategory(
            name=category,
            description="",
            impact_level=3,
        )

    return await service.get_statistics(
        UserId(current_user.id),
        category=category_obj,
        period_days=period_days,
    )


@router.get("/{behavior_id}/related", response_model=list[BehaviorResponse])
async def find_related_behaviors(
    behavior_id: int,
    time_window_minutes: int = 60,
    service: BehaviorService = Depends(get_behavior_service),
    current_user: User = Depends(get_current_active_user),
):
    """関連する行動を検索"""
    return await service.find_related_behaviors(
        BehaviorId(behavior_id),
        time_window_minutes=time_window_minutes,
    )


@router.get("/emotion-triggered", response_model=list[BehaviorResponse])
async def find_emotion_triggered_behaviors(
    emotion: str,
    threshold: float = 0.5,
    service: BehaviorService = Depends(get_behavior_service),
    current_user: User = Depends(get_current_active_user),
):
    """感情によってトリガーされた行動を検索"""
    return await service.find_emotion_triggered_behaviors(
        UserId(current_user.id),
        emotion,
        threshold=threshold,
    )


@router.post("/goals", response_model=BehaviorGoalResponse)
async def create_behavior_goal(
    data: BehaviorGoalCreate,
    service: BehaviorService = Depends(get_behavior_service),
    current_user: User = Depends(get_current_active_user),
):
    """行動目標を作成"""
    category = BehaviorCategory(
        name=data.category,
        description=data.category_description,
        impact_level=data.impact_level,
    )

    goal = BehaviorGoal(
        category=category,
        target_value=data.target_value,
        target_unit=data.target_unit,
        frequency_target=data.frequency_target,
        start_date=data.start_date,
        end_date=data.end_date,
        progress=0.0,
        status="not_started",
    )

    return await service.update_behavior_goal(
        UserId(current_user.id),
        goal,
    )


@router.get("/triggers", response_model=list[BehaviorTriggerResponse])
async def identify_behavior_triggers(
    category: str,
    days: int = 30,
    service: BehaviorService = Depends(get_behavior_service),
    current_user: User = Depends(get_current_active_user),
):
    """行動のトリガーを特定"""
    category_obj = BehaviorCategory(
        name=category,
        description="",
        impact_level=3,
    )

    return await service.identify_triggers(
        UserId(current_user.id),
        category_obj,
        days=days,
    )
