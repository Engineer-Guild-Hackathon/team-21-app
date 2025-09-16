"""
アバター・称号システムのAPIエンドポイント

ゲーミフィケーション要素として、ユーザーの学習意欲向上を促進
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ...core.security import get_current_user
from ...domain.models.avatar import (
    Avatar,
    AvatarPart,
    Title,
    UserAvatar,
    UserStats,
    UserTitle,
)
from ...domain.models.user import User
from ...domain.schemas.avatar import (
    AvatarChangeRequest,
    AvatarPartResponse,
    AvatarResponse,
    TitleChangeRequest,
    TitleResponse,
    UserAvatarResponse,
    UserProfileResponse,
    UserStatsResponse,
    UserStatsUpdate,
    UserTitleResponse,
)
from ...infrastructure.database import get_db

router = APIRouter()


@router.get("/avatars", response_model=List[AvatarResponse])
async def get_avatars(
    category: Optional[str] = Query(None, description="アバターカテゴリでフィルタ"),
    rarity: Optional[str] = Query(None, description="レアリティでフィルタ"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """利用可能なアバター一覧を取得"""

    stmt = select(Avatar).where(Avatar.is_active)

    if category:
        stmt = stmt.where(Avatar.category == category)
    if rarity:
        stmt = stmt.where(Avatar.rarity == rarity)

    stmt = stmt.order_by(Avatar.sort_order, Avatar.created_at)

    result = await db.execute(stmt)
    avatars = result.scalars().all()

    return avatars


@router.get("/avatars/parts", response_model=List[AvatarPartResponse])
async def get_avatar_parts(
    part_type: Optional[str] = Query(None, description="パーツタイプでフィルタ"),
    rarity: Optional[str] = Query(None, description="レアリティでフィルタ"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """利用可能なアバターパーツ一覧を取得"""

    stmt = select(AvatarPart).where(AvatarPart.is_active)

    if part_type:
        stmt = stmt.where(AvatarPart.part_type == part_type)
    if rarity:
        stmt = stmt.where(AvatarPart.rarity == rarity)

    stmt = stmt.order_by(AvatarPart.sort_order, AvatarPart.created_at)

    result = await db.execute(stmt)
    parts = result.scalars().all()

    return parts


@router.get("/titles", response_model=List[TitleResponse])
async def get_titles(
    category: Optional[str] = Query(None, description="称号カテゴリでフィルタ"),
    rarity: Optional[str] = Query(None, description="レアリティでフィルタ"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """利用可能な称号一覧を取得"""

    stmt = select(Title).where(Title.is_active)

    if category:
        stmt = stmt.where(Title.category == category)
    if rarity:
        stmt = stmt.where(Title.rarity == rarity)

    stmt = stmt.order_by(Title.sort_order, Title.created_at)

    result = await db.execute(stmt)
    titles = result.scalars().all()

    return titles


@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """ユーザーのプロフィール情報を取得（アバター・称号情報含む）"""

    # 現在のアバターを取得
    current_avatar_stmt = (
        select(UserAvatar)
        .options(selectinload(UserAvatar.avatar))
        .where(and_(UserAvatar.user_id == current_user.id, UserAvatar.is_current))
    )
    current_avatar_result = await db.execute(current_avatar_stmt)
    current_avatar = current_avatar_result.scalar_one_or_none()

    # 現在の称号を取得
    current_title_stmt = (
        select(UserTitle)
        .options(selectinload(UserTitle.title))
        .where(and_(UserTitle.user_id == current_user.id, UserTitle.is_current))
    )
    current_title_result = await db.execute(current_title_stmt)
    current_title = current_title_result.scalar_one_or_none()

    # 所持しているアバター一覧を取得
    owned_avatars_stmt = (
        select(UserAvatar)
        .options(selectinload(UserAvatar.avatar))
        .where(UserAvatar.user_id == current_user.id)
    )
    owned_avatars_result = await db.execute(owned_avatars_stmt)
    owned_avatars = owned_avatars_result.scalars().all()

    # 初回アクセス時: デフォルトアバター（ひよこ）を自動付与して現在アバターに設定
    if not owned_avatars:
        # デフォルト候補を検索（名前で優先的に探し、なければ最も若い並び順のアクティブアバター）
        default_avatar_stmt = (
            select(Avatar)
            .where(
                and_(
                    Avatar.is_active,
                    Avatar.name.in_(["ひよこ", "Chick", "ひよこ(初期)"]),
                )
            )
            .order_by(Avatar.sort_order, Avatar.created_at)
        )
        default_avatar_result = await db.execute(default_avatar_stmt)
        default_avatar = default_avatar_result.scalars().first()

        if not default_avatar:
            # 既存がない場合はデフォルトアバターを作成
            default_avatar = Avatar(
                name="ひよこ",
                description="初期アバター",
                image_url="/avatars/chick.png",
                category="character",
                rarity="common",
                is_active=True,
                sort_order=0,
            )
            db.add(default_avatar)
            await db.commit()
            await db.refresh(default_avatar)

        if default_avatar:
            new_user_avatar = UserAvatar(
                user_id=current_user.id,
                avatar_id=default_avatar.id,
                is_current=True,
            )
            db.add(new_user_avatar)
            # 同一トランザクションで avatar 作成→付与を確定させる
            await db.commit()

            # 再取得してレスポンスへ反映
            owned_avatars_stmt = (
                select(UserAvatar)
                .options(selectinload(UserAvatar.avatar))
                .where(UserAvatar.user_id == current_user.id)
            )
            owned_avatars_result = await db.execute(owned_avatars_stmt)
            owned_avatars = owned_avatars_result.scalars().all()

            current_avatar_stmt = (
                select(UserAvatar)
                .options(selectinload(UserAvatar.avatar))
                .where(
                    and_(UserAvatar.user_id == current_user.id, UserAvatar.is_current)
                )
            )
            current_avatar_result = await db.execute(current_avatar_stmt)
            current_avatar = current_avatar_result.scalar_one_or_none()

    # 所持している称号一覧を取得
    owned_titles_stmt = (
        select(UserTitle)
        .options(selectinload(UserTitle.title))
        .where(UserTitle.user_id == current_user.id)
    )
    owned_titles_result = await db.execute(owned_titles_stmt)
    owned_titles = owned_titles_result.scalars().all()

    # ユーザー統計を取得
    stats_stmt = select(UserStats).where(UserStats.user_id == current_user.id)
    stats_result = await db.execute(stats_stmt)
    stats = stats_result.scalar_one_or_none()

    if not stats:
        # 統計が存在しない場合は作成
        stats = UserStats(user_id=current_user.id)
        db.add(stats)
        await db.commit()
        await db.refresh(stats)

    # レベル計算（簡易版：総クエスト完了数に基づく）
    level = min(100, 1 + (stats.total_quests_completed // 10))

    return UserProfileResponse(
        id=current_user.id,
        name=current_user.full_name or "Unknown",
        email=current_user.email,
        role=current_user.role,
        current_avatar=current_avatar,
        current_title=current_title,
        available_avatars=owned_avatars,
        available_titles=owned_titles,
        stats=stats,
        level=level,
    )


@router.post("/avatar/change", response_model=UserAvatarResponse)
async def change_avatar(
    request: AvatarChangeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """現在のアバターを変更"""

    # アバターが存在し、ユーザーが所持しているか確認
    user_avatar_stmt = (
        select(UserAvatar)
        .options(selectinload(UserAvatar.avatar))
        .where(
            and_(
                UserAvatar.user_id == current_user.id,
                UserAvatar.avatar_id == request.avatar_id,
            )
        )
    )
    user_avatar_result = await db.execute(user_avatar_stmt)
    user_avatar = user_avatar_result.scalar_one_or_none()

    if not user_avatar:
        raise HTTPException(
            status_code=404, detail="アバターが見つからないか、まだ解除されていません"
        )

    # 現在のアバターを非アクティブにする
    current_avatar_stmt = select(UserAvatar).where(
        and_(UserAvatar.user_id == current_user.id, UserAvatar.is_current)
    )
    current_avatar_result = await db.execute(current_avatar_stmt)
    current_avatar = current_avatar_result.scalar_one_or_none()

    if current_avatar:
        current_avatar.is_current = False

    # 新しいアバターをアクティブにする
    user_avatar.is_current = True

    await db.commit()
    await db.refresh(user_avatar)

    return user_avatar


@router.post("/title/change", response_model=UserTitleResponse)
async def change_title(
    request: TitleChangeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """現在の称号を変更"""

    # 称号が存在し、ユーザーが所持しているか確認
    user_title_stmt = (
        select(UserTitle)
        .options(selectinload(UserTitle.title))
        .where(
            and_(
                UserTitle.user_id == current_user.id,
                UserTitle.title_id == request.title_id,
            )
        )
    )
    user_title_result = await db.execute(user_title_stmt)
    user_title = user_title_result.scalar_one_or_none()

    if not user_title:
        raise HTTPException(
            status_code=404, detail="称号が見つからないか、まだ獲得していません"
        )

    # 現在の称号を非アクティブにする
    current_title_stmt = select(UserTitle).where(
        and_(UserTitle.user_id == current_user.id, UserTitle.is_current)
    )
    current_title_result = await db.execute(current_title_stmt)
    current_title = current_title_result.scalar_one_or_none()

    if current_title:
        current_title.is_current = False

    # 新しい称号をアクティブにする
    user_title.is_current = True

    await db.commit()
    await db.refresh(user_title)

    return user_title


@router.get("/stats", response_model=UserStatsResponse)
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """ユーザーの統計情報を取得"""

    stmt = select(UserStats).where(UserStats.user_id == current_user.id)
    result = await db.execute(stmt)
    stats = result.scalar_one_or_none()

    if not stats:
        # 統計が存在しない場合は作成
        stats = UserStats(user_id=current_user.id)
        db.add(stats)
        await db.commit()
        await db.refresh(stats)

    return stats


@router.put("/stats", response_model=UserStatsResponse)
async def update_user_stats(
    stats_update: UserStatsUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """ユーザーの統計情報を更新"""

    stmt = select(UserStats).where(UserStats.user_id == current_user.id)
    result = await db.execute(stmt)
    stats = result.scalar_one_or_none()

    if not stats:
        stats = UserStats(user_id=current_user.id)
        db.add(stats)

    # 更新データを適用
    update_data = stats_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(stats, field, value)

    await db.commit()
    await db.refresh(stats)

    # アバター・称号の自動解除をチェック
    await check_and_unlock_achievements(current_user.id, db)

    return stats


async def check_and_unlock_achievements(user_id: int, db: AsyncSession):
    """ユーザーの統計に基づいてアバター・称号の自動解除をチェック"""

    # ユーザー統計を取得
    stats_stmt = select(UserStats).where(UserStats.user_id == user_id)
    stats_result = await db.execute(stats_stmt)
    stats = stats_result.scalar_one_or_none()

    if not stats:
        return

    # アバター解除チェック
    avatars_stmt = select(Avatar).where(
        and_(
            Avatar.is_active,
            Avatar.unlock_condition_type.isnot(None),
            Avatar.unlock_condition_value.isnot(None),
        )
    )
    avatars_result = await db.execute(avatars_stmt)
    avatars = avatars_result.scalars().all()

    for avatar in avatars:
        # 既に所持しているかチェック
        existing_stmt = select(UserAvatar).where(
            and_(UserAvatar.user_id == user_id, UserAvatar.avatar_id == avatar.id)
        )
        existing_result = await db.execute(existing_stmt)
        existing = existing_result.scalar_one_or_none()

        if existing:
            continue

        # 解除条件をチェック
        should_unlock = False
        if avatar.unlock_condition_type == "quest_count":
            should_unlock = (
                stats.total_quests_completed >= avatar.unlock_condition_value
            )
        elif avatar.unlock_condition_type == "streak_days":
            should_unlock = stats.max_streak_days >= avatar.unlock_condition_value
        elif avatar.unlock_condition_type == "level_reach":
            level = 1 + (stats.total_quests_completed // 10)
            should_unlock = level >= avatar.unlock_condition_value

        if should_unlock:
            # アバターを解除
            user_avatar = UserAvatar(
                user_id=user_id,
                avatar_id=avatar.id,
                is_current=False,  # 最初は非アクティブ
            )
            db.add(user_avatar)
            stats.total_avatars_unlocked += 1

    # 称号解除チェック
    titles_stmt = select(Title).where(
        and_(
            Title.is_active,
            Title.unlock_condition_type.isnot(None),
            Title.unlock_condition_value.isnot(None),
        )
    )
    titles_result = await db.execute(titles_stmt)
    titles = titles_result.scalars().all()

    for title in titles:
        # 既に所持しているかチェック
        existing_stmt = select(UserTitle).where(
            and_(UserTitle.user_id == user_id, UserTitle.title_id == title.id)
        )
        existing_result = await db.execute(existing_stmt)
        existing = existing_result.scalar_one_or_none()

        if existing:
            continue

        # 解除条件をチェック
        should_unlock = False
        if title.unlock_condition_type == "quest_count":
            should_unlock = stats.total_quests_completed >= title.unlock_condition_value
        elif title.unlock_condition_type == "streak_days":
            should_unlock = stats.max_streak_days >= title.unlock_condition_value
        elif title.unlock_condition_type == "skill_level":
            # 全スキルの平均レベルをチェック
            avg_skill_level = (
                stats.grit_level
                + stats.collaboration_level
                + stats.self_regulation_level
                + stats.emotional_intelligence_level
            ) / 4
            should_unlock = avg_skill_level >= title.unlock_condition_value

        if should_unlock:
            # 称号を獲得
            user_title = UserTitle(
                user_id=user_id,
                title_id=title.id,
                is_current=False,  # 最初は非アクティブ
            )
            db.add(user_title)
            stats.total_titles_earned += 1

    await db.commit()
