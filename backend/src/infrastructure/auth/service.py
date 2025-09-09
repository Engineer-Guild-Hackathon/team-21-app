from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ...core.security import create_access_token, verify_password
from ...domain.models.user import User


class AuthService:
    """認証サービス

    認証に関連する操作を提供するインフラストラクチャサービス。
    ドメインロジックは含まず、認証に特化した機能のみを提供する。
    """

    def __init__(self, db: Session):
        self.db = db

    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """ユーザー認証

        Args:
            email: ユーザーのメールアドレス
            password: 平文のパスワード

        Returns:
            認証成功時はUserオブジェクト、失敗時はNone
        """
        user = self.db.query(User).filter(User.email == email).first()
        if not user or not verify_password(password, str(user.hashed_password)):
            return None

        # 認証成功時の処理（監査ログ等）
        user.last_login = datetime.utcnow()
        self.db.commit()

        return user

    async def create_user_token(
        self, user: User, expires_delta: Optional[timedelta] = None
    ) -> dict:
        """アクセストークンの生成

        Args:
            user: トークンを生成するユーザー
            expires_delta: トークンの有効期限

        Returns:
            アクセストークン情報を含む辞書
        """
        access_token = create_access_token(
            subject=user.email, expires_delta=expires_delta
        )

        return {"access_token": access_token, "token_type": "bearer"}

    async def revoke_token(self, token: str) -> bool:
        """トークンの無効化

        Args:
            token: 無効化するトークン

        Returns:
            無効化が成功したかどうか
        """
        # トークンブラックリストへの追加等の実装
        return True
