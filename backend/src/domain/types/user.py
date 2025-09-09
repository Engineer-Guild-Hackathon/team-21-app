from dataclasses import dataclass
from datetime import datetime
from typing import NewType

# 値オブジェクト
UserId = NewType("UserId", int)
Email = NewType("Email", str)
PasswordHash = NewType("PasswordHash", str)


@dataclass(frozen=True)
class UserCredentials:
    """ユーザー認証情報

    パスワード関連の値オブジェクト。
    不変性を保証するためにfrozenなデータクラスとして実装。
    """

    email: Email
    password_hash: PasswordHash


@dataclass(frozen=True)
class UserProfile:
    """ユーザープロフィール情報

    ユーザーの表示用情報を表す値オブジェクト。
    """

    full_name: str
    avatar_url: str | None = None
    bio: str | None = None


@dataclass(frozen=True)
class UserStatus:
    """ユーザーステータス情報

    ユーザーの状態を表す値オブジェクト。
    """

    is_active: bool
    is_verified: bool
    last_login: datetime | None
    created_at: datetime
    updated_at: datetime
