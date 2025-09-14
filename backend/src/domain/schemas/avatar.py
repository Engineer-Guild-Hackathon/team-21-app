"""
アバター・称号システムのPydanticスキーマ

APIのリクエスト・レスポンス用のデータ検証とシリアライゼーション
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# アバター関連スキーマ
class AvatarBase(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    image_url: Optional[str] = Field(None, max_length=500)
    category: str = Field(..., max_length=50)
    rarity: str = Field(default="common", max_length=20)
    unlock_condition_type: Optional[str] = Field(None, max_length=50)
    unlock_condition_value: Optional[int] = None
    sort_order: int = Field(default=0)


class AvatarCreate(AvatarBase):
    pass


class AvatarResponse(AvatarBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# アバターパーツ関連スキーマ
class AvatarPartBase(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    image_url: Optional[str] = Field(None, max_length=500)
    part_type: str = Field(..., max_length=50)
    rarity: str = Field(default="common", max_length=20)
    unlock_condition_type: Optional[str] = Field(None, max_length=50)
    unlock_condition_value: Optional[int] = None
    sort_order: int = Field(default=0)


class AvatarPartCreate(AvatarPartBase):
    pass


class AvatarPartResponse(AvatarPartBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# 称号関連スキーマ
class TitleBase(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    icon_url: Optional[str] = Field(None, max_length=500)
    category: str = Field(..., max_length=50)
    rarity: str = Field(default="common", max_length=20)
    unlock_condition_type: str = Field(..., max_length=50)
    unlock_condition_value: int
    unlock_condition_description: Optional[str] = None
    sort_order: int = Field(default=0)


class TitleCreate(TitleBase):
    pass


class TitleResponse(TitleBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ユーザーアバター関連スキーマ
class UserAvatarBase(BaseModel):
    avatar_id: int
    is_current: bool = False


class UserAvatarCreate(UserAvatarBase):
    pass


class UserAvatarResponse(UserAvatarBase):
    id: int
    user_id: int
    unlocked_at: datetime
    avatar: AvatarResponse

    class Config:
        from_attributes = True


# ユーザー称号関連スキーマ
class UserTitleBase(BaseModel):
    title_id: int
    is_current: bool = False


class UserTitleCreate(UserTitleBase):
    pass


class UserTitleResponse(UserTitleBase):
    id: int
    user_id: int
    unlocked_at: datetime
    title: TitleResponse

    class Config:
        from_attributes = True


# ユーザー統計関連スキーマ
class UserStatsBase(BaseModel):
    total_quests_completed: int = 0
    daily_quests_completed: int = 0
    current_streak_days: int = 0
    max_streak_days: int = 0
    total_learning_time_minutes: int = 0
    total_sessions: int = 0
    grit_level: float = 1.0
    collaboration_level: float = 1.0
    self_regulation_level: float = 1.0
    emotional_intelligence_level: float = 1.0
    total_titles_earned: int = 0
    total_avatars_unlocked: int = 0


class UserStatsResponse(UserStatsBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserStatsUpdate(BaseModel):
    total_quests_completed: Optional[int] = None
    daily_quests_completed: Optional[int] = None
    current_streak_days: Optional[int] = None
    max_streak_days: Optional[int] = None
    total_learning_time_minutes: Optional[int] = None
    total_sessions: Optional[int] = None
    grit_level: Optional[float] = None
    collaboration_level: Optional[float] = None
    self_regulation_level: Optional[float] = None
    emotional_intelligence_level: Optional[float] = None
    total_titles_earned: Optional[int] = None
    total_avatars_unlocked: Optional[int] = None


# ユーザープロフィール関連スキーマ（アバター・称号情報を含む）
class UserProfileResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    current_avatar: Optional[UserAvatarResponse] = None
    current_title: Optional[UserTitleResponse] = None
    available_avatars: List[UserAvatarResponse] = []
    available_titles: List[UserTitleResponse] = []
    stats: UserStatsResponse
    level: int = 1  # 計算されたレベル

    class Config:
        from_attributes = True


# アバター変更リクエスト
class AvatarChangeRequest(BaseModel):
    avatar_id: int


class TitleChangeRequest(BaseModel):
    title_id: int


# 獲得通知用スキーマ
class AchievementNotification(BaseModel):
    type: str  # "avatar_unlocked" or "title_earned"
    name: str
    description: str
    image_url: Optional[str] = None
    rarity: str = "common"
    unlocked_at: datetime
