"""
クエスト管理APIエンドポイント

非認知能力を高める学習クエストの実装
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ...core.security import get_current_user
from ...domain.models.avatar import UserStats
from ...domain.models.quest import (
    Quest,
    QuestProgress,
    QuestReward,
    QuestSession,
    QuestStatus,
    QuestType,
)
from ...domain.models.user import User
from ...domain.schemas.quest import (
    QuestListResponse,
    QuestProgressCreate,
    QuestProgressListResponse,
    QuestProgressResponse,
    QuestProgressUpdate,
    QuestProgressWithQuest,
    QuestRecommendationResponse,
    QuestResponse,
    QuestRewardResponse,
    QuestSessionCreate,
    QuestSessionResponse,
    QuestStatsResponse,
)
from ...infrastructure.database import get_db
from .avatars import check_and_unlock_achievements

router = APIRouter()


async def update_user_stats_on_quest_completion(user_id: int, quest, db: AsyncSession):
    """クエスト完了時にユーザー統計を更新"""
    # ユーザー統計を取得または作成
    stats_stmt = select(UserStats).where(UserStats.user_id == user_id)
    stats_result = await db.execute(stats_stmt)
    stats = stats_result.scalar_one_or_none()

    if not stats:
        stats = UserStats(user_id=user_id)
        db.add(stats)

    # 完了クエスト数を増加
    stats.total_quests_completed += 1

    # 日次クエストの場合
    if quest.is_daily:
        stats.daily_quests_completed += 1
        # 連続日数を更新（簡易実装）
        stats.current_streak_days += 1
        stats.max_streak_days = max(stats.max_streak_days, stats.current_streak_days)

    # 学習時間を増加（推定）
    stats.total_learning_time_minutes += quest.estimated_duration
    stats.total_sessions += 1


@router.get("/", response_model=QuestListResponse)
async def get_quests(
    page: int = Query(1, ge=1, description="ページ番号"),
    size: int = Query(10, ge=1, le=100, description="ページサイズ"),
    quest_type: Optional[QuestType] = Query(
        None, description="クエストタイプでフィルタ"
    ),
    difficulty: Optional[str] = Query(None, description="難易度でフィルタ"),
    is_daily: Optional[bool] = Query(None, description="日次クエストでフィルタ"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """利用可能なクエスト一覧を取得"""

    # クエリ構築
    stmt = select(Quest).where(Quest.is_active)

    # フィルタ適用
    if quest_type:
        stmt = stmt.where(Quest.quest_type == quest_type)
    if difficulty:
        stmt = stmt.where(Quest.difficulty == difficulty)
    if is_daily is not None:
        stmt = stmt.where(Quest.is_daily == is_daily)

    # レベル制限（学生のみ）
    if current_user.role == "student":
        stmt = stmt.where(Quest.required_level <= 1)  # 仮のレベル制限

    # ソートとページネーション
    stmt = stmt.order_by(Quest.sort_order, Quest.created_at)

    # 総数を取得
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = await db.scalar(count_stmt)

    offset = (page - 1) * size
    stmt = stmt.offset(offset).limit(size)
    quests = await db.execute(stmt)
    quest_list = quests.scalars().all()

    return QuestListResponse(
        quests=[QuestResponse.from_orm(q) for q in quest_list],
        total=total,
        page=page,
        size=size,
    )


@router.get("/recommended", response_model=QuestRecommendationResponse)
async def get_recommended_quests(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """ユーザーに推奨されるクエストを取得"""

    # ユーザーの過去の進捗を分析
    progress_stmt = select(QuestProgress).where(
        and_(
            QuestProgress.user_id == current_user.id,
            QuestProgress.status == QuestStatus.COMPLETED,
        )
    )
    completed_progress = await db.execute(progress_stmt)
    completed_quests = completed_progress.scalars().all()

    # 完了したクエストタイプを分析
    completed_types = set()
    for progress in completed_quests:
        quest = await db.get(Quest, progress.quest_id)
        if quest:
            completed_types.add(quest.quest_type)

    # 推奨ロジック（完了していないタイプを優先）
    all_types = set(QuestType)
    recommended_types = (
        all_types - completed_types if completed_types else [QuestType.DAILY_LOG]
    )

    # 推奨クエストを取得
    recommended_stmt = (
        select(Quest)
        .where(
            Quest.is_active,
            Quest.quest_type.in_(recommended_types),
            Quest.required_level <= 1,  # 仮のレベル制限
        )
        .limit(3)
    )

    recommended_quests = await db.execute(recommended_stmt)
    quest_list = recommended_quests.scalars().all()

    return QuestRecommendationResponse(
        recommended_quests=[QuestResponse.from_orm(q) for q in quest_list],
        reason="未完了のクエストタイプを推奨",
        based_on={
            "completed_types": list(completed_types),
            "user_level": 1,
        },
    )


@router.post("/{quest_id}/start", response_model=QuestProgressResponse)
async def start_quest(
    quest_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """クエストを開始"""

    # クエストの存在確認
    quest = await db.get(Quest, quest_id)
    if not quest:
        raise HTTPException(status_code=404, detail="クエストが見つかりません")

    if not quest.is_active:
        raise HTTPException(status_code=400, detail="このクエストは利用できません")

    # 既存の進捗確認
    existing_progress = await db.execute(
        select(QuestProgress).where(
            and_(
                QuestProgress.user_id == current_user.id,
                QuestProgress.quest_id == quest_id,
                QuestProgress.status.in_(
                    [QuestStatus.NOT_STARTED, QuestStatus.IN_PROGRESS]
                ),
            )
        )
    )
    progress = existing_progress.scalar_one_or_none()

    if progress:
        # 既存の進捗を更新
        progress.status = QuestStatus.IN_PROGRESS
        progress.started_date = datetime.utcnow()
    else:
        # 新しい進捗を作成
        progress_data = QuestProgressCreate(
            user_id=current_user.id,
            quest_id=quest_id,
            status=QuestStatus.IN_PROGRESS,
            started_date=datetime.utcnow(),
        )
        progress = QuestProgress(**progress_data.dict())
        db.add(progress)

    await db.commit()
    await db.refresh(progress)

    return QuestProgressResponse.from_orm(progress)


@router.put("/progress/{progress_id}", response_model=QuestProgressResponse)
async def update_quest_progress(
    progress_id: int,
    progress_update: QuestProgressUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """クエスト進捗を更新"""

    # 進捗の存在確認と権限チェック
    progress = await db.get(QuestProgress, progress_id)
    if not progress:
        raise HTTPException(status_code=404, detail="進捗が見つかりません")

    if progress.user_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="この進捗を更新する権限がありません"
        )

    # 進捗更新
    update_data = progress_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(progress, field, value)

    # 完了時の処理
    if progress_update.status == QuestStatus.COMPLETED:
        progress.completed_date = datetime.utcnow()
        progress.progress_percentage = 100.0

        # 報酬を付与
        quest = await db.get(Quest, progress.quest_id)
        if quest:
            reward = QuestReward(
                user_id=current_user.id,
                quest_id=quest.id,
                reward_type="experience",
                reward_value=quest.experience_points,
            )
            db.add(reward)
            # クエスト完了時に即時反映させるため自動受取
            reward.is_claimed = True
            reward.claimed_at = datetime.utcnow()

            if quest.coins > 0:
                coin_reward = QuestReward(
                    user_id=current_user.id,
                    quest_id=quest.id,
                    reward_type="coins",
                    reward_value=quest.coins,
                )
                db.add(coin_reward)
                coin_reward.is_claimed = True
                coin_reward.claimed_at = datetime.utcnow()

            if quest.badge_id:
                badge_reward = QuestReward(
                    user_id=current_user.id,
                    quest_id=quest.id,
                    reward_type="badge",
                    reward_value=1,
                    reward_data={"badge_id": quest.badge_id},
                )
                db.add(badge_reward)

            # 統計情報を更新
            await update_user_stats_on_quest_completion(current_user.id, quest, db)

            # クエスト完了に伴う称号の解除チェック（初クエストなどを即時付与）
            await check_and_unlock_achievements(current_user.id, db)

    await db.commit()
    await db.refresh(progress)

    return QuestProgressResponse.from_orm(progress)


@router.get("/my-progress", response_model=QuestProgressListResponse)
async def get_my_progress(
    page: int = Query(1, ge=1, description="ページ番号"),
    size: int = Query(10, ge=1, le=100, description="ページサイズ"),
    status: Optional[QuestStatus] = Query(None, description="ステータスでフィルタ"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """自分のクエスト進捗一覧を取得"""

    # クエリ構築
    stmt = (
        select(QuestProgress)
        .options(selectinload(QuestProgress.quest))
        .where(QuestProgress.user_id == current_user.id)
    )

    if status:
        stmt = stmt.where(QuestProgress.status == status)

    # ソートとページネーション
    stmt = stmt.order_by(desc(QuestProgress.updated_at))

    # 総数を取得
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = await db.scalar(count_stmt)

    offset = (page - 1) * size
    stmt = stmt.offset(offset).limit(size)
    progress_list = await db.execute(stmt)
    progress_items = progress_list.scalars().all()

    return QuestProgressListResponse(
        progress=[QuestProgressWithQuest.from_orm(p) for p in progress_items],
        total=total,
        page=page,
        size=size,
    )


@router.get("/stats", response_model=QuestStatsResponse)
async def get_quest_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """クエスト統計情報を取得"""

    # 基本的な統計
    total_quests = await db.scalar(select(func.count(Quest.id)).where(Quest.is_active))

    completed_count = await db.scalar(
        select(func.count(QuestProgress.id)).where(
            and_(
                QuestProgress.user_id == current_user.id,
                QuestProgress.status == QuestStatus.COMPLETED,
            )
        )
    )

    in_progress_count = await db.scalar(
        select(func.count(QuestProgress.id)).where(
            and_(
                QuestProgress.user_id == current_user.id,
                QuestProgress.status == QuestStatus.IN_PROGRESS,
            )
        )
    )

    # 獲得経験値とコイン
    total_experience = (
        await db.scalar(
            select(func.sum(QuestReward.reward_value)).where(
                and_(
                    QuestReward.user_id == current_user.id,
                    QuestReward.reward_type == "experience",
                    QuestReward.is_claimed,
                )
            )
        )
        or 0
    )

    total_coins = (
        await db.scalar(
            select(func.sum(QuestReward.reward_value)).where(
                and_(
                    QuestReward.user_id == current_user.id,
                    QuestReward.reward_type == "coins",
                    QuestReward.is_claimed,
                )
            )
        )
        or 0
    )

    # 連続達成日数
    streak_days = (
        await db.scalar(
            select(func.max(QuestProgress.streak_count)).where(
                QuestProgress.user_id == current_user.id
            )
        )
        or 0
    )

    # お気に入りのクエストタイプ
    favorite_type_query = await db.execute(
        select(Quest.quest_type, func.count(QuestProgress.id).label("count"))
        .select_from(Quest)
        .join(QuestProgress)
        .where(
            and_(
                QuestProgress.user_id == current_user.id,
                QuestProgress.status == QuestStatus.COMPLETED,
            )
        )
        .group_by(Quest.quest_type)
        .order_by(desc("count"))
        .limit(1)
    )
    favorite_result = favorite_type_query.first()
    favorite_quest_type = favorite_result[0] if favorite_result else None

    return QuestStatsResponse(
        total_quests=total_quests,
        completed_quests=completed_count,
        in_progress_quests=in_progress_count,
        total_experience=total_experience,
        total_coins=total_coins,
        streak_days=streak_days,
        favorite_quest_type=favorite_quest_type,
    )


@router.post("/sessions", response_model=QuestSessionResponse)
async def create_quest_session(
    session_data: QuestSessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """クエストセッションを作成"""

    # クエストの存在確認
    quest = await db.get(Quest, session_data.quest_id)
    if not quest:
        raise HTTPException(status_code=404, detail="クエストが見つかりません")

    # セッション作成
    session = QuestSession(
        user_id=current_user.id,
        quest_id=session_data.quest_id,
        session_data=session_data.session_data,
        is_active=session_data.is_active,
        started_at=session_data.started_at,
    )

    db.add(session)
    await db.commit()
    await db.refresh(session)

    return QuestSessionResponse.from_orm(session)


@router.get("/rewards", response_model=List[QuestRewardResponse])
async def get_my_rewards(
    is_claimed: Optional[bool] = Query(None, description="受取済みでフィルタ"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """自分の報酬一覧を取得"""

    stmt = select(QuestReward).where(QuestReward.user_id == current_user.id)

    if is_claimed is not None:
        stmt = stmt.where(QuestReward.is_claimed == is_claimed)

    stmt = stmt.order_by(desc(QuestReward.created_at))
    rewards = await db.execute(stmt)
    reward_list = rewards.scalars().all()

    return [QuestRewardResponse.from_orm(r) for r in reward_list]


@router.post("/rewards/{reward_id}/claim", response_model=QuestRewardResponse)
async def claim_reward(
    reward_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """報酬を受け取る"""

    reward = await db.get(QuestReward, reward_id)
    if not reward:
        raise HTTPException(status_code=404, detail="報酬が見つかりません")

    if reward.user_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="この報酬を受け取る権限がありません"
        )

    if reward.is_claimed:
        raise HTTPException(status_code=400, detail="この報酬は既に受け取られています")

    reward.is_claimed = True
    reward.claimed_at = datetime.utcnow()

    await db.commit()
    await db.refresh(reward)

    return QuestRewardResponse.from_orm(reward)
