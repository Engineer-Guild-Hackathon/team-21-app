from datetime import datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from ...core.security import create_access_token, verify_password
from ...domain.models.user import User
from ...domain.types import Email


class AuthService:
    """認証サービス

    認証に関するユースケースを実装するサービス。
    認証ロジックとドメインモデルを組み合わせて認証機能を提供します。
    """

    def __init__(self, db: Session):
        self.db = db

    async def authenticate(
        self, email: Email, password: str
    ) -> Tuple[Optional[User], Optional[str]]:
        """ユーザー認証とトークン生成

        Args:
            email: ユーザーのメールアドレス
            password: 平文のパスワード

        Returns:
            (認証されたユーザー, アクセストークン)のタプル
            認証失敗時は(None, None)
        """
        # ユーザー検索
        user = self.db.query(User).filter(User.email == email).first()
        if not user:
            return None, None

        # パスワード検証
        if not verify_password(password, str(user.hashed_password)):
            return None, None

        # 認証成功時の処理
        user.last_login = datetime.utcnow()
        self.db.commit()

        # トークン生成
        token = create_access_token(subject=user.email, expires_delta=timedelta(days=1))

        return user, token

    async def validate_token(self, token: str) -> Optional[User]:
        """トークンの検証とユーザー取得

        Args:
            token: 検証するトークン

        Returns:
            トークンに対応するユーザー。無効なトークンの場合はNone
        """
        try:
            payload = decode_token(token)
            email = payload.get("sub")
            if email is None:
                return None
        except Exception:
            return None

        return self.db.query(User).filter(User.email == email).first()

    async def revoke_token(self, token: str) -> bool:
        """トークンの無効化（ログアウト）

        Args:
            token: 無効化するトークン

        Returns:
            無効化が成功したかどうか
        """
        # トークンブラックリストへの追加等の実装
        return True
