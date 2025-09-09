from abc import abstractmethod
from datetime import datetime
from typing import List, Optional

from ..models.behavior import Behavior
from ..types.behavior import BehaviorCategory, BehaviorId, BehaviorPattern, UserId
from .base import Repository


class BehaviorRepository(Repository[Behavior]):
    """行動データリポジトリのインターフェース

    行動分析データの永続化操作を定義します。
    実装はインフラストラクチャ層で提供されます。
    """

    @abstractmethod
    async def find_by_user(
        self,
        user_id: UserId,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        category: Optional[str] = None,
    ) -> List[Behavior]:
        """ユーザーIDによる行動データの検索"""
        pass

    @abstractmethod
    async def find_by_category(
        self,
        category: BehaviorCategory,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Behavior]:
        """カテゴリによる行動データの検索"""
        pass

    @abstractmethod
    async def find_patterns(
        self, user_id: UserId, period_days: int = 30
    ) -> List[BehaviorPattern]:
        """行動パターンの検索"""
        pass

    @abstractmethod
    async def get_user_statistics(
        self,
        user_id: UserId,
        category: Optional[BehaviorCategory] = None,
        period_days: int = 7,
    ) -> dict:
        """ユーザーの行動統計を取得"""
        pass

    @abstractmethod
    async def find_correlated_behaviors(
        self,
        behavior_id: BehaviorId,
        threshold: float = 0.5,
        time_window_minutes: int = 60,
    ) -> List[Behavior]:
        """相関のある行動を検索"""
        pass

    @abstractmethod
    async def find_by_emotion(
        self,
        user_id: UserId,
        emotion: str,
        threshold: float = 0.5,
        time_window_minutes: int = 60,
    ) -> List[Behavior]:
        """感情状態に関連する行動を検索"""
        pass
