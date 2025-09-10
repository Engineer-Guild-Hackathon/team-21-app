from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ...infrastructure.database import Base


class Emotion(Base):
    """感情記録モデル"""

    __tablename__ = "emotion_records"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    emotion_type: Mapped[str] = mapped_column()
    intensity: Mapped[float] = mapped_column()
    context: Mapped[Optional[str]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # リレーションシップ
    user = relationship("User", back_populates="emotion_records")
