from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ...infrastructure.database import Base


class User(Base):
    """ユーザーモデル"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
    full_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    role: Mapped[str] = mapped_column(String, nullable=False, default="student")
    class_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("classes.id"), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # リレーションシップ
    emotion_records = relationship(
        "Emotion",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    enrolled_class = relationship(
        "Class", back_populates="students", foreign_keys=[class_id]
    )
    taught_classes = relationship(
        "Class", back_populates="teacher", foreign_keys="Class.teacher_id"
    )
    learning_progress = relationship(
        "LearningProgress",
        back_populates="student",
        cascade="all, delete-orphan",
    )
    quest_progresses = relationship(
        "QuestProgress",
        back_populates="user",
        cascade="all, delete-orphan",
    )


# リレーション解決のために明示的に読み込む（テスト時の単独import対策）
from .classroom import Class, LearningProgress  # noqa: E402,F401
from .emotion import Emotion  # noqa: E402,F401
from .quest import QuestProgress  # noqa: E402,F401
