from abc import abstractmethod
from typing import List, Optional

from ..models.user import User
from ..types.user import Email, UserId
from .base import Repository


class UserRepository(Repository[User]):
    """ユーザーリポジトリのインターフェース

    ユーザーエンティティの永続化操作を定義します。
    実装はインフラストラクチャ層で提供されます。
    """

    @abstractmethod
    async def find_by_email(self, email: Email) -> Optional[User]:
        """メールアドレスによるユーザー検索"""
        pass

    @abstractmethod
    async def find_active_users(self) -> List[User]:
        """アクティブなユーザー一覧の取得"""
        pass

    @abstractmethod
    async def update_last_login(self, user_id: UserId) -> None:
        """最終ログイン日時の更新"""
        pass
