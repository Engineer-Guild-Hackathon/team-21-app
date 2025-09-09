from abc import abstractmethod
from datetime import datetime
from typing import List, Optional

from ..models.emotion import Emotion
from ..types.emotion import UserId
from .base import Repository


class EmotionRepository(Repository[Emotion]):
    """感情データリポジトリのインターフェース

    感情分析データの永続化操作を定義します。
    実装はインフラストラクチャ層で提供されます。
    """

    @abstractmethod
    async def find_by_user(
        self,
        user_id: UserId,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Emotion]:
        """ユーザーIDによる感情データの検索"""
        pass

    @abstractmethod
    async def find_latest_by_user(
        self, user_id: UserId, limit: int = 1
    ) -> List[Emotion]:
        """ユーザーの最新の感情データを取得"""
        pass

    @abstractmethod
    async def find_by_emotion(
        self, emotion: str, threshold: float = 0.5
    ) -> List[Emotion]:
        """特定の感情カテゴリによる検索"""
        pass

    @abstractmethod
    async def get_user_trends(
        self, user_id: UserId, period_days: int = 7
    ) -> List[dict]:
        """ユーザーの感情トレンドを取得"""
        pass
