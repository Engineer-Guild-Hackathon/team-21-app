from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class Repository(Generic[T], ABC):
    """リポジトリの基底クラス

    全てのリポジトリ実装の基底となるインターフェース。
    CRUDの基本操作を定義します。
    """

    @abstractmethod
    async def create(self, entity: T) -> T:
        """エンティティの作成"""
        pass

    @abstractmethod
    async def read(self, id: any) -> Optional[T]:
        """IDによるエンティティの取得"""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """エンティティの更新"""
        pass

    @abstractmethod
    async def delete(self, id: any) -> bool:
        """エンティティの削除"""
        pass

    @abstractmethod
    async def list(self, skip: int = 0, limit: int = 100) -> List[T]:
        """エンティティ一覧の取得"""
        pass
