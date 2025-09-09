from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from ...domain.models.emotion import Emotion
from ...domain.repositories.emotion_repository import EmotionRepository
from ...domain.types.emotion import EmotionId, UserId


class SQLAlchemyEmotionRepository(EmotionRepository):
    """SQLAlchemyを使用した感情データリポジトリの実装"""

    def __init__(self, db: Session):
        self.db = db

    async def create(self, entity: Emotion) -> Emotion:
        """感情データの作成"""
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    async def read(self, id: EmotionId) -> Optional[Emotion]:
        """IDによる感情データの取得"""
        return self.db.query(Emotion).filter(Emotion.id == id).first()

    async def update(self, entity: Emotion) -> Emotion:
        """感情データの更新"""
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    async def delete(self, id: EmotionId) -> bool:
        """感情データの削除"""
        emotion = await self.read(id)
        if not emotion:
            return False
        self.db.delete(emotion)
        self.db.commit()
        return True

    async def list(self, skip: int = 0, limit: int = 100) -> List[Emotion]:
        """感情データ一覧の取得"""
        return self.db.query(Emotion).offset(skip).limit(limit).all()

    async def find_by_user(
        self,
        user_id: UserId,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Emotion]:
        """ユーザーIDによる感情データの検索"""
        query = self.db.query(Emotion).filter(Emotion.user_id == user_id)

        if start_time:
            query = query.filter(Emotion.timestamp >= start_time)
        if end_time:
            query = query.filter(Emotion.timestamp <= end_time)

        return query.order_by(Emotion.timestamp.desc()).all()

    async def find_latest_by_user(
        self, user_id: UserId, limit: int = 1
    ) -> List[Emotion]:
        """ユーザーの最新の感情データを取得"""
        return (
            self.db.query(Emotion)
            .filter(Emotion.user_id == user_id)
            .order_by(Emotion.timestamp.desc())
            .limit(limit)
            .all()
        )

    async def find_by_emotion(
        self, emotion: str, threshold: float = 0.5
    ) -> List[Emotion]:
        """特定の感情カテゴリによる検索"""
        return (
            self.db.query(Emotion)
            .filter(Emotion.scores[emotion].astext.cast(float) >= threshold)
            .order_by(Emotion.timestamp.desc())
            .all()
        )

    async def get_user_trends(
        self, user_id: UserId, period_days: int = 7
    ) -> List[dict]:
        """ユーザーの感情トレンドを取得"""
        start_date = datetime.utcnow() - timedelta(days=period_days)

        # 日ごとの平均感情スコアを計算
        daily_scores = (
            self.db.query(
                func.date_trunc("day", Emotion.timestamp).label("date"),
                func.avg(Emotion.scores["happiness"].astext.cast(float)).label(
                    "happiness"
                ),
                func.avg(Emotion.scores["sadness"].astext.cast(float)).label("sadness"),
                func.avg(Emotion.scores["anger"].astext.cast(float)).label("anger"),
                func.avg(Emotion.scores["fear"].astext.cast(float)).label("fear"),
            )
            .filter(Emotion.user_id == user_id, Emotion.timestamp >= start_date)
            .group_by(func.date_trunc("day", Emotion.timestamp))
            .order_by(func.date_trunc("day", Emotion.timestamp))
            .all()
        )

        return [
            {
                "date": score.date,
                "emotions": {
                    "happiness": float(score.happiness),
                    "sadness": float(score.sadness),
                    "anger": float(score.anger),
                    "fear": float(score.fear),
                },
            }
            for score in daily_scores
        ]
