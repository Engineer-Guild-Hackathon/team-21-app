from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ...infrastructure.database import Base


class ActionType(str, Enum):
    """行動の種類"""

    PROBLEM_START = "problem_start"  # 問題開始
    PROBLEM_SUBMIT = "problem_submit"  # 回答提出
    PROBLEM_RETRY = "problem_retry"  # 再挑戦
    HINT_REQUEST = "hint_request"  # ヒント要求
    COLLABORATION = "collaboration"  # 協力行動
    GIVE_UP = "give_up"  # 諦める
    REFLECTION = "reflection"  # 振り返り


class BehaviorRecord(Base):
    """行動記録モデル"""

    __tablename__ = "behavior_records"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    problem_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("problems.id"), nullable=True
    )
    action_type: Mapped[ActionType] = mapped_column()
    start_time: Mapped[datetime] = mapped_column()
    end_time: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    attempt_count: Mapped[int] = mapped_column(default=1)
    success: Mapped[Optional[bool]] = mapped_column(nullable=True)
    approach_description: Mapped[Optional[str]] = mapped_column(nullable=True)
    emotion_state: Mapped[Optional[str]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # リレーションシップ
    user = relationship("User", back_populates="behavior_records")
    problem = relationship("Problem", back_populates="behavior_records")
