"""
アバター・称号システムのデータベースモデル

ゲーミフィケーション要素として、ユーザーの学習意欲向上を促進
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.sql import func

from ...infrastructure.database import Base


class Avatar(Base):
    """アバター情報"""

    __tablename__ = "avatars"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text)
    image_url = Column(String(500))
    category = Column(String(50), nullable=False)  # animal, character, robot, etc.
    rarity = Column(
        String(20), nullable=False, default="common"
    )  # common, rare, epic, legendary
    unlock_condition_type = Column(
        String(50)
    )  # quest_complete, level_reach, special_achievement
    unlock_condition_value = Column(Integer)  # 必要な値（クエスト数、レベルなど）
    is_active = Column(Boolean, default=True)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class AvatarPart(Base):
    """アバターパーツ（帽子、眼鏡、アクセサリーなど）"""

    __tablename__ = "avatar_parts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    image_url = Column(String(500))
    part_type = Column(String(50), nullable=False)  # hat, glasses, accessory, etc.
    rarity = Column(String(20), nullable=False, default="common")
    unlock_condition_type = Column(String(50))
    unlock_condition_value = Column(Integer)
    is_active = Column(Boolean, default=True)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class UserAvatar(Base):
    """ユーザーのアバター設定"""

    __tablename__ = "user_avatars"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    avatar_id = Column(Integer, ForeignKey("avatars.id"), nullable=False)
    is_current = Column(Boolean, default=False)  # 現在使用中のアバター
    unlocked_at = Column(DateTime(timezone=True), server_default=func.now())

    # リレーション（後で設定）


class UserAvatarPart(Base):
    """ユーザーのアバターパーツ所持状況"""

    __tablename__ = "user_avatar_parts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    part_id = Column(Integer, ForeignKey("avatar_parts.id"), nullable=False)
    is_equipped = Column(Boolean, default=False)  # 現在装備中かどうか
    unlocked_at = Column(DateTime(timezone=True), server_default=func.now())

    # リレーション（後で設定）


class Title(Base):
    """称号情報"""

    __tablename__ = "titles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text)
    icon_url = Column(String(500))
    category = Column(
        String(50), nullable=False
    )  # learning, quest, cooperation, special
    rarity = Column(String(20), nullable=False, default="common")
    unlock_condition_type = Column(
        String(50), nullable=False
    )  # quest_count, streak_days, skill_level
    unlock_condition_value = Column(Integer, nullable=False)
    unlock_condition_description = Column(Text)  # 獲得条件の説明
    is_active = Column(Boolean, default=True)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class UserTitle(Base):
    """ユーザーの称号所持状況"""

    __tablename__ = "user_titles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title_id = Column(Integer, ForeignKey("titles.id"), nullable=False)
    is_current = Column(Boolean, default=False)  # 現在使用中の称号
    unlocked_at = Column(DateTime(timezone=True), server_default=func.now())

    # リレーション（後で設定）


class UserStats(Base):
    """ユーザーの統計情報（称号・アバター獲得判定用）"""

    __tablename__ = "user_stats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)

    # クエスト関連統計
    total_quests_completed = Column(Integer, default=0)
    daily_quests_completed = Column(Integer, default=0)
    current_streak_days = Column(Integer, default=0)
    max_streak_days = Column(Integer, default=0)

    # 学習関連統計
    total_learning_time_minutes = Column(Integer, default=0)
    total_sessions = Column(Integer, default=0)

    # スキルレベル
    grit_level = Column(Float, default=1.0)
    collaboration_level = Column(Float, default=1.0)
    self_regulation_level = Column(Float, default=1.0)
    emotional_intelligence_level = Column(Float, default=1.0)

    # 称号・アバター関連
    total_titles_earned = Column(Integer, default=0)
    total_avatars_unlocked = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # リレーション（後で設定）
