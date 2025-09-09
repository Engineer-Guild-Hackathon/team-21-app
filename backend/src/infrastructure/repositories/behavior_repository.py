from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from ...domain.models.behavior import Behavior
from ...domain.models.emotion import Emotion
from ...domain.repositories.behavior_repository import BehaviorRepository
from ...domain.types.behavior import (
    BehaviorCategory,
    BehaviorId,
    BehaviorPattern,
    UserId,
)


class SQLAlchemyBehaviorRepository(BehaviorRepository):
    """SQLAlchemyを使用した行動データリポジトリの実装"""

    def __init__(self, db: Session):
        self.db = db

    async def create(self, entity: Behavior) -> Behavior:
        """行動データの作成"""
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    async def read(self, id: BehaviorId) -> Optional[Behavior]:
        """IDによる行動データの取得"""
        return self.db.query(Behavior).filter(Behavior.id == id).first()

    async def update(self, entity: Behavior) -> Behavior:
        """行動データの更新"""
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    async def delete(self, id: BehaviorId) -> bool:
        """行動データの削除"""
        behavior = await self.read(id)
        if not behavior:
            return False
        self.db.delete(behavior)
        self.db.commit()
        return True

    async def list(self, skip: int = 0, limit: int = 100) -> List[Behavior]:
        """行動データ一覧の取得"""
        return self.db.query(Behavior).offset(skip).limit(limit).all()

    async def find_by_user(
        self,
        user_id: UserId,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        category: Optional[str] = None,
    ) -> List[Behavior]:
        """ユーザーIDによる行動データの検索"""
        query = self.db.query(Behavior).filter(Behavior.user_id == user_id)

        if start_time:
            query = query.filter(Behavior.timestamp >= start_time)
        if end_time:
            query = query.filter(Behavior.timestamp <= end_time)
        if category:
            query = query.filter(Behavior.category == category)

        return query.order_by(Behavior.timestamp.desc()).all()

    async def find_by_category(
        self,
        category: BehaviorCategory,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Behavior]:
        """カテゴリによる行動データの検索"""
        query = self.db.query(Behavior).filter(Behavior.category == category.name)

        if start_time:
            query = query.filter(Behavior.timestamp >= start_time)
        if end_time:
            query = query.filter(Behavior.timestamp <= end_time)

        return query.order_by(Behavior.timestamp.desc()).all()

    async def find_patterns(
        self, user_id: UserId, period_days: int = 30
    ) -> List[BehaviorPattern]:
        """行動パターンの検索"""
        start_date = datetime.utcnow() - timedelta(days=period_days)

        # 行動の集計クエリ
        patterns = (
            self.db.query(
                Behavior.category,
                func.count(Behavior.id).label("frequency"),
                func.avg(Behavior.duration).label("average_duration"),
                func.array_agg(func.extract("hour", Behavior.timestamp)).label("hours"),
                func.array_agg(func.extract("dow", Behavior.timestamp)).label("days"),
            )
            .filter(Behavior.user_id == user_id, Behavior.timestamp >= start_date)
            .group_by(Behavior.category)
            .all()
        )

        return [self._convert_to_pattern(pattern) for pattern in patterns]

    def _convert_to_pattern(self, pattern) -> BehaviorPattern:
        """SQLクエリ結果をBehaviorPatternに変換"""
        # 時間帯の分類
        hours = pattern.hours
        time_of_day = []
        if any(5 <= h < 12 for h in hours):
            time_of_day.append("morning")
        if any(12 <= h < 17 for h in hours):
            time_of_day.append("afternoon")
        if any(17 <= h < 22 for h in hours):
            time_of_day.append("evening")
        if any(h >= 22 or h < 5 for h in hours):
            time_of_day.append("night")

        # 曜日の変換
        days = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        days_of_week = [days[int(d)] for d in pattern.days]

        # 一貫性スコアの計算（0-1のスケール）
        time_consistency = len(set(hours)) / 24  # 時間の一貫性
        day_consistency = len(set(pattern.days)) / 7  # 曜日の一貫性
        consistency_score = (time_consistency + day_consistency) / 2

        return BehaviorPattern(
            category=BehaviorCategory(
                name=pattern.category,
                description="",  # データベースから取得する必要があります
                impact_level=3,  # デフォルト値
            ),
            frequency=pattern.frequency,
            average_duration=float(pattern.average_duration),
            time_of_day=time_of_day,
            days_of_week=list(set(days_of_week)),
            consistency_score=consistency_score,
        )

    async def get_user_statistics(
        self,
        user_id: UserId,
        category: Optional[BehaviorCategory] = None,
        period_days: int = 7,
    ) -> dict:
        """ユーザーの行動統計を取得"""
        start_date = datetime.utcnow() - timedelta(days=period_days)
        query = self.db.query(Behavior).filter(
            Behavior.user_id == user_id, Behavior.timestamp >= start_date
        )

        if category:
            query = query.filter(Behavior.category == category.name)

        behaviors = query.all()

        if not behaviors:
            return {
                "total_count": 0,
                "total_duration": 0,
                "average_duration": 0,
                "categories": {},
                "time_distribution": {},
            }

        # 基本統計の計算
        total_count = len(behaviors)
        total_duration = sum(b.duration for b in behaviors)
        average_duration = total_duration / total_count

        # カテゴリごとの集計
        categories = {}
        for behavior in behaviors:
            cat = behavior.category
            if cat not in categories:
                categories[cat] = {
                    "count": 0,
                    "total_duration": 0,
                    "average_duration": 0,
                }
            categories[cat]["count"] += 1
            categories[cat]["total_duration"] += behavior.duration
            categories[cat]["average_duration"] = (
                categories[cat]["total_duration"] / categories[cat]["count"]
            )

        # 時間帯ごとの分布
        time_distribution = {
            "morning": 0,
            "afternoon": 0,
            "evening": 0,
            "night": 0,
        }
        for behavior in behaviors:
            hour = behavior.timestamp.hour
            if 5 <= hour < 12:
                time_distribution["morning"] += 1
            elif 12 <= hour < 17:
                time_distribution["afternoon"] += 1
            elif 17 <= hour < 22:
                time_distribution["evening"] += 1
            else:
                time_distribution["night"] += 1

        return {
            "total_count": total_count,
            "total_duration": total_duration,
            "average_duration": average_duration,
            "categories": categories,
            "time_distribution": time_distribution,
        }

    async def find_correlated_behaviors(
        self,
        behavior_id: BehaviorId,
        threshold: float = 0.5,
        time_window_minutes: int = 60,
    ) -> List[Behavior]:
        """相関のある行動を検索"""
        base_behavior = await self.read(behavior_id)
        if not base_behavior:
            return []

        window_start = base_behavior.timestamp - timedelta(minutes=time_window_minutes)
        window_end = base_behavior.timestamp + timedelta(minutes=time_window_minutes)

        return (
            self.db.query(Behavior)
            .filter(
                Behavior.id != behavior_id,
                Behavior.user_id == base_behavior.user_id,
                Behavior.timestamp.between(window_start, window_end),
            )
            .order_by(Behavior.timestamp)
            .all()
        )

    async def find_by_emotion(
        self,
        user_id: UserId,
        emotion: str,
        threshold: float = 0.5,
        time_window_minutes: int = 60,
    ) -> List[Behavior]:
        """感情状態に関連する行動を検索"""
        # 感情データの取得
        emotion_records = (
            self.db.query(Emotion)
            .filter(
                Emotion.user_id == user_id,
                Emotion.scores[emotion].astext.cast(float) >= threshold,
            )
            .all()
        )

        behaviors = []
        for emotion_record in emotion_records:
            window_start = emotion_record.timestamp - timedelta(
                minutes=time_window_minutes
            )
            window_end = emotion_record.timestamp + timedelta(
                minutes=time_window_minutes
            )

            # 時間枠内の行動を検索
            time_window_behaviors = (
                self.db.query(Behavior)
                .filter(
                    Behavior.user_id == user_id,
                    Behavior.timestamp.between(window_start, window_end),
                )
                .order_by(Behavior.timestamp)
                .all()
            )
            behaviors.extend(time_window_behaviors)

        return list(set(behaviors))  # 重複を除去
