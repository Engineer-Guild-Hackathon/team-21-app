from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    """ユーザーの基本スキーマ"""

    email: EmailStr
    full_name: str
    is_active: bool = True
    is_verified: bool = False


class UserCreate(UserBase):
    """ユーザー作成スキーマ"""

    password: str
    role: str = "student"
    class_id: Optional[str] = None
    terms_accepted: bool = False


class UserUpdate(UserBase):
    """ユーザー更新スキーマ"""

    password: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None


class UserResponse(UserBase):
    """ユーザーレスポンススキーマ"""

    id: int
    role: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
