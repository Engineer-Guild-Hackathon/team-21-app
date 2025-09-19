"""
クエスト関連のSQLAlchemyモデル

非認知能力を高める学習クエストの実装
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import JSON, Boolean, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from ...infrastructure.database import Base


class QuestType(str, Enum):
    """クエストタイプの定義"""

    DAILY_LOG = "daily_log"  # 今日の冒険ログ
    PLANT_CARE = "plant_care"  # 魔法の種を育てよう
    STORY_CREATION = "story_creation"  # お話の森
    COLLABORATION = "collaboration"  # 協力！謎解きチャットルーム


class QuestDifficulty(str, Enum):
    """クエスト難易度"""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestStatus(str, Enum):
    """クエストステータス"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    LOCKED = "locked"


class Quest(Base):
    """クエスト基本情報テーブル"""

    __tablename__ = "quests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    quest_type: Mapped[QuestType] = mapped_column(SQLEnum(QuestType), nullable=False)
    difficulty: Mapped[QuestDifficulty] = mapped_column(
        SQLEnum(QuestDifficulty), nullable=False
    )

    # クエスト設定
    target_skill: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # 対象非認知能力
    estimated_duration: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # 推定所要時間（分）
    required_level: Mapped[int] = mapped_column(Integer, default=1)  # 必要レベル

    # クエスト設定（JSON形式で柔軟な設定を保存）
    quest_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # 報酬設定
    experience_points: Mapped[int] = mapped_column(Integer, default=100)
    coins: Mapped[int] = mapped_column(Integer, default=50)
    badge_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # システム設定
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_daily: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # 日次クエストかどうか
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # リレーション
    quest_progresses: Mapped[List["QuestProgress"]] = relationship(
        "QuestProgress", back_populates="quest"
    )


class QuestProgress(Base):
    """クエスト進捗テーブル"""

    __tablename__ = "quest_progresses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    quest_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("quests.id"), nullable=False
    )

    # 進捗情報
    status: Mapped[QuestStatus] = mapped_column(
        SQLEnum(QuestStatus), default=QuestStatus.NOT_STARTED
    )
    progress_percentage: Mapped[float] = mapped_column(Float, default=0.0)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=1)

    # クエスト固有の進捗データ（JSON形式）
    quest_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # 日次クエスト用
    started_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    streak_count: Mapped[int] = mapped_column(Integer, default=0)  # 連続達成回数

    # 評価・フィードバック
    self_evaluation: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # 1-5の自己評価
    teacher_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # リレーション
    user: Mapped["User"] = relationship("User", back_populates="quest_progresses")
    quest: Mapped["Quest"] = relationship("Quest", back_populates="quest_progresses")


# リレーション解決のために明示的に読み込む（テスト時の単独import対策）
from .user import User  # noqa: E402,F401


class QuestSession(Base):
    """クエストセッションテーブル（セッション単位での進捗追跡）"""

    __tablename__ = "quest_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    quest_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("quests.id"), nullable=False
    )

    # セッション情報
    session_data: Mapped[dict] = mapped_column(
        JSON, nullable=False
    )  # セッション固有のデータ
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # リレーション
    user: Mapped["User"] = relationship("User")
    quest: Mapped["Quest"] = relationship("Quest")


class QuestReward(Base):
    """クエスト報酬テーブル"""

    __tablename__ = "quest_rewards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    quest_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("quests.id"), nullable=False
    )

    # 報酬内容
    reward_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # experience, coins, badge, item
    reward_value: Mapped[int] = mapped_column(Integer, nullable=False)
    reward_data: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # 追加データ（バッジ情報など）

    is_claimed: Mapped[bool] = mapped_column(Boolean, default=False)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # リレーション
    user: Mapped["User"] = relationship("User")
    quest: Mapped["Quest"] = relationship("Quest")
