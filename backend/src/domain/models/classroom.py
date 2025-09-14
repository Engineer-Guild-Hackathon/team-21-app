from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ...infrastructure.database import Base


class Class(Base):
    """クラスモデル"""

    __tablename__ = "classes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    class_id: Mapped[str] = mapped_column(
        String, unique=True, index=True
    )  # ABC-123形式
    name: Mapped[str] = mapped_column(String, nullable=False)  # クラス名
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    teacher_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # リレーションシップ
    teacher = relationship(
        "User", back_populates="taught_classes", foreign_keys=[teacher_id]
    )
    students = relationship(
        "User", back_populates="enrolled_class", foreign_keys="User.class_id"
    )
    learning_progress = relationship(
        "LearningProgress",
        back_populates="class_",
        cascade="all, delete-orphan",
    )


class LearningProgress(Base):
    """学習進捗モデル"""

    __tablename__ = "learning_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    student_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    class_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("classes.id"), nullable=False
    )

    # 非認知能力スコア
    grit_score: Mapped[float] = mapped_column(nullable=False, default=0.0)  # やり抜く力
    collaboration_score: Mapped[float] = mapped_column(
        nullable=False, default=0.0
    )  # 協調性
    self_regulation_score: Mapped[float] = mapped_column(
        nullable=False, default=0.0
    )  # 自己調整
    emotional_intelligence_score: Mapped[float] = mapped_column(
        nullable=False, default=0.0
    )  # 感情知性

    # 学習進捗
    quests_completed: Mapped[int] = mapped_column(Integer, default=0)  # クエスト完了数
    total_learning_time: Mapped[int] = mapped_column(
        Integer, default=0
    )  # 総学習時間（分）
    retry_count: Mapped[int] = mapped_column(Integer, default=0)  # リトライ回数

    # タイムスタンプ
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # リレーションシップ
    student = relationship(
        "User", back_populates="learning_progress", foreign_keys=[student_id]
    )
    class_ = relationship(
        "Class", back_populates="learning_progress", foreign_keys=[class_id]
    )


# リレーション解決のために明示的に読み込む（テスト時の単独import対策）
from .user import User  # noqa: E402,F401
