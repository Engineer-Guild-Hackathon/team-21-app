from typing import List, Optional

from ...core.security import get_password_hash
from ...domain.models.user import User
from ...domain.repositories.user_repository import UserRepository
from ...domain.types.user import Email, UserId, UserProfile, UserStatus


class UserService:
    """ユーザー管理のユースケース実装

    ユーザーに関する操作を実装するサービス。
    リポジトリを使用してユーザーの永続化を行います。
    """

    def __init__(self, repository: UserRepository):
        self.repository = repository

    async def create_user(self, email: Email, password: str, full_name: str) -> User:
        """新規ユーザーの作成"""
        # ビジネスルール: メールアドレスの重複チェック
        if await self.repository.find_by_email(email):
            raise ValueError("Email already registered")

        # ユーザーの作成
        user = User(
            email=email,
            hashed_password=get_password_hash(password),
            full_name=full_name,
            is_active=True,
            is_verified=False,
        )

        return await self.repository.create(user)

    async def get_user(self, user_id: UserId) -> Optional[User]:
        """ユーザーの取得"""
        return await self.repository.read(user_id)

    async def get_user_by_email(self, email: Email) -> Optional[User]:
        """メールアドレスによるユーザー取得"""
        return await self.repository.find_by_email(email)

    async def list_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """ユーザー一覧の取得"""
        return await self.repository.list(skip, limit)

    async def update_profile(
        self, user_id: UserId, profile: UserProfile
    ) -> Optional[User]:
        """ユーザープロフィールの更新"""
        user = await self.repository.read(user_id)
        if not user:
            return None

        # プロフィール情報の更新
        user.full_name = profile.full_name
        user.avatar_url = profile.avatar_url
        user.bio = profile.bio

        return await self.repository.update(user)

    async def update_status(
        self, user_id: UserId, status: UserStatus
    ) -> Optional[User]:
        """ユーザーステータスの更新"""
        user = await self.repository.read(user_id)
        if not user:
            return None

        # ステータス情報の更新
        user.is_active = status.is_active
        user.is_verified = status.is_verified

        return await self.repository.update(user)

    async def delete_user(self, user_id: UserId) -> bool:
        """ユーザーの削除"""
        return await self.repository.delete(user_id)

    async def update_last_login(self, user_id: UserId) -> None:
        """最終ログイン日時の更新"""
        await self.repository.update_last_login(user_id)
