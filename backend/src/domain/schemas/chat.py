"""
チャット関連のPydanticスキーマ
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ChatMessageCreate(BaseModel):
    """チャットメッセージ作成スキーマ"""

    content: str
    role: str  # 'user' or 'assistant'


class ChatMessageResponse(BaseModel):
    """チャットメッセージレスポンススキーマ"""

    id: int
    session_id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSessionCreate(BaseModel):
    """チャットセッション作成スキーマ"""

    title: Optional[str] = None


class ChatSessionResponse(BaseModel):
    """チャットセッションレスポンススキーマ"""

    id: int
    user_id: int
    title: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    messages: List[ChatMessageResponse] = []

    class Config:
        from_attributes = True


class ChatSessionListResponse(BaseModel):
    """チャットセッション一覧レスポンススキーマ"""

    id: int
    title: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    message_count: int

    class Config:
        from_attributes = True
