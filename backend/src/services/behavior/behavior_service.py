from datetime import datetime, timedelta
from typing import List, Optional

from ...domain.models.behavior import Behavior
from ...domain.repositories.behavior_repository import BehaviorRepository
from ...domain.types.behavior import (
    BehaviorCategory,
    BehaviorGoal,
    BehaviorId,
    BehaviorMetrics,
    BehaviorPattern,
    BehaviorTrigger,
    UserId,
)


class BehaviorService:
    """行動分析のユースケース実装

    行動データの管理と分析を行うサービス。
    リポジトリを使用して行動データの永続化を行います。
    """

    def __init__(self, repository: BehaviorRepository):
        self.repository = repository

    async def record_behavior(
        self,
        user_id: UserId,
        category: BehaviorCategory,
        metrics: List[BehaviorMetrics],
        duration: float,
        notes: Optional[str] = None,
    ) -> Behavior:
        """行動データの記録"""
        behavior = Behavior(
            user_id=user_id,
            category=category.name,
            metrics=metrics,
            duration=duration,
            timestamp=datetime.utcnow(),
            notes=notes,
        )

        return await self.repository.create(behavior)

    async def get_user_behaviors(
        self,
        user_id: UserId,
        category: Optional[BehaviorCategory] = None,
        days: int = 7,
    ) -> List[Behavior]:
        """ユーザーの行動履歴を取得"""
        start_time = datetime.utcnow() - timedelta(days=days)
        return await self.repository.find_by_user(
            user_id, start_time=start_time, category=category.name if category else None
        )

    async def analyze_patterns(
        self, user_id: UserId, period_days: int = 30
    ) -> List[BehaviorPattern]:
        """行動パターンの分析"""
        return await self.repository.find_patterns(user_id, period_days=period_days)

    async def get_statistics(
        self,
        user_id: UserId,
        category: Optional[BehaviorCategory] = None,
        period_days: int = 7,
    ) -> dict:
        """行動統計の取得"""
        return await self.repository.get_user_statistics(
            user_id, category=category, period_days=period_days
        )

    async def find_related_behaviors(
        self, behavior_id: BehaviorId, time_window_minutes: int = 60
    ) -> List[Behavior]:
        """関連する行動の検索"""
        return await self.repository.find_correlated_behaviors(
            behavior_id, time_window_minutes=time_window_minutes
        )

    async def find_emotion_triggered_behaviors(
        self, user_id: UserId, emotion: str, threshold: float = 0.5
    ) -> List[Behavior]:
        """感情によってトリガーされた行動の検索"""
        return await self.repository.find_by_emotion(
            user_id, emotion, threshold=threshold
        )

    async def update_behavior_goal(
        self, user_id: UserId, goal: BehaviorGoal
    ) -> BehaviorGoal:
        """行動目標の更新"""
        # 目標の進捗を計算
        behaviors = await self.repository.find_by_user(
            user_id,
            start_time=goal.start_date,
            end_time=goal.end_date,
            category=goal.category.name,
        )

        if not behaviors:
            return goal

        # 目標の種類に応じた進捗計算
        total_value = sum(
            next((m.value for m in b.metrics if m.metric_name == goal.target_unit), 0)
            for b in behaviors
        )
        progress = min(total_value / goal.target_value, 1.0)

        # 状態の更新
        if progress >= 1.0:
            status = "completed"
        elif progress > 0:
            status = "in_progress"
        else:
            status = "not_started"

        return BehaviorGoal(
            category=goal.category,
            target_value=goal.target_value,
            target_unit=goal.target_unit,
            frequency_target=goal.frequency_target,
            start_date=goal.start_date,
            end_date=goal.end_date,
            progress=progress,
            status=status,
        )

    async def identify_triggers(
        self, user_id: UserId, category: BehaviorCategory, days: int = 30
    ) -> List[BehaviorTrigger]:
        """行動のトリガーを特定"""
        # 感情トリガーの分析
        emotion_triggers = []
        emotions = ["happiness", "sadness", "anger", "fear"]

        for emotion in emotions:
            behaviors = await self.repository.find_by_emotion(
                user_id,
                emotion,
                threshold=0.7,  # 高い確信度の感情のみを考慮
                time_window_minutes=30,  # 感情の直後の行動を検索
            )

            # このカテゴリの行動のみをフィルタリング
            category_behaviors = [b for b in behaviors if b.category == category.name]

            if category_behaviors:
                frequency = len(category_behaviors)
                confidence = frequency / len(behaviors) if behaviors else 0

                if confidence > 0.3:  # 一定以上の確信度があるものだけを抽出
                    emotion_triggers.append(
                        BehaviorTrigger(
                            trigger_type="emotion",
                            condition={"emotion": emotion, "threshold": 0.7},
                            confidence=confidence,
                            last_triggered=max(b.timestamp for b in category_behaviors),
                            frequency=frequency,
                        )
                    )

        # 時間トリガーの分析
        behaviors = await self.repository.find_by_user(
            user_id,
            start_time=datetime.utcnow() - timedelta(days=days),
            category=category.name,
        )

        if behaviors:
            # 時間帯ごとの発生頻度を分析
            time_distribution = {}
            for behavior in behaviors:
                hour = behavior.timestamp.hour
                if hour not in time_distribution:
                    time_distribution[hour] = 0
                time_distribution[hour] += 1

            # 最も頻度の高い時間帯を特定
            max_hour = max(time_distribution.items(), key=lambda x: x[1])
            time_confidence = max_hour[1] / len(behaviors)

            if time_confidence > 0.3:
                time_triggers = [
                    BehaviorTrigger(
                        trigger_type="time",
                        condition={"hour": max_hour[0]},
                        confidence=time_confidence,
                        last_triggered=max(
                            b.timestamp
                            for b in behaviors
                            if b.timestamp.hour == max_hour[0]
                        ),
                        frequency=max_hour[1],
                    )
                ]
            else:
                time_triggers = []
        else:
            time_triggers = []

        return emotion_triggers + time_triggers
