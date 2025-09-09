from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from ...domain.models.user import User
from ...domain.repositories.user_repository import UserRepository
from ...domain.types.user import Email, UserId


class SQLAlchemyUserRepository(UserRepository):
    """SQLAlchemyを使用したユーザーリポジトリの実装"""

    def __init__(self, db: Session):
        self.db = db

    async def create(self, entity: User) -> User:
        """ユーザーの作成"""
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    async def read(self, id: UserId) -> Optional[User]:
        """IDによるユーザー取得"""
        return self.db.query(User).filter(User.id == id).first()

    async def update(self, entity: User) -> User:
        """ユーザーの更新"""
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    async def delete(self, id: UserId) -> bool:
        """ユーザーの削除"""
        user = await self.read(id)
        if not user:
            return False
        self.db.delete(user)
        self.db.commit()
        return True

    async def list(self, skip: int = 0, limit: int = 100) -> List[User]:
        """ユーザー一覧の取得"""
        return self.db.query(User).offset(skip).limit(limit).all()

    async def find_by_email(self, email: Email) -> Optional[User]:
        """メールアドレスによるユーザー検索"""
        return self.db.query(User).filter(User.email == email).first()

    async def find_active_users(self) -> List[User]:
        """アクティブなユーザー一覧の取得"""
        return self.db.query(User).filter(User.is_active == True).all()

    async def update_last_login(self, user_id: UserId) -> None:
        """最終ログイン日時の更新"""
        user = await self.read(user_id)
        if user:
            user.last_login = datetime.utcnow()
            self.db.commit()
