from datetime import datetime, timedelta
from typing import List, Optional

from ...domain.models.emotion import Emotion
from ...domain.repositories.emotion_repository import EmotionRepository
from ...domain.types.emotion import EmotionAnalysis, EmotionTrend, UserId


class EmotionService:
    """感情分析のユースケース実装

    感情データの管理と分析を行うサービス。
    リポジトリを使用して感情データの永続化を行います。
    """

    def __init__(self, repository: EmotionRepository):
        self.repository = repository

    async def record_emotion(
        self, user_id: UserId, analysis: EmotionAnalysis
    ) -> Emotion:
        """感情分析結果の記録"""
        emotion = Emotion(
            user_id=user_id,
            scores=analysis.scores,
            dominant_emotion=analysis.dominant_emotion,
            timestamp=analysis.timestamp,
            source_type=analysis.source_type,
            source_content=analysis.source_content,
        )

        return await self.repository.create(emotion)

    async def get_user_emotions(self, user_id: UserId, days: int = 7) -> List[Emotion]:
        """ユーザーの感情履歴を取得"""
        start_time = datetime.utcnow() - timedelta(days=days)
        return await self.repository.find_by_user(user_id, start_time=start_time)

    async def get_latest_emotion(self, user_id: UserId) -> Optional[Emotion]:
        """ユーザーの最新の感情状態を取得"""
        emotions = await self.repository.find_latest_by_user(user_id)
        return emotions[0] if emotions else None

    async def analyze_trend(
        self, user_id: UserId, period_days: int = 7
    ) -> EmotionTrend:
        """感情トレンドの分析"""
        trends = await self.repository.get_user_trends(user_id, period_days=period_days)

        if not trends:
            return None

        # トレンドの方向性を判定
        start_scores = trends[0]["emotions"]
        end_scores = trends[-1]["emotions"]

        # 主要感情の変化を計算
        changes = []
        for emotion in ["happiness", "sadness", "anger", "fear"]:
            start = start_scores[emotion]
            end = end_scores[emotion]
            change = end - start
            changes.append(change)

        # 全体的なトレンドを判定
        avg_change = sum(changes) / len(changes)
        if abs(avg_change) < 0.1:
            trend_direction = "stable"
        elif avg_change > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"

        return EmotionTrend(
            start_time=trends[0]["date"],
            end_time=trends[-1]["date"],
            emotion_changes=changes,
            trend_direction=trend_direction,
            confidence=0.8,  # 信頼度は実際のデータに基づいて計算する必要があります
        )

    async def find_similar_emotions(
        self, emotion: str, threshold: float = 0.5
    ) -> List[Emotion]:
        """類似した感情状態のデータを検索"""
        return await self.repository.find_by_emotion(emotion, threshold=threshold)
