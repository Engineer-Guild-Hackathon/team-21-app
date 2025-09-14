"""
クエスト関連のPydanticスキーマ

非認知能力を高める学習クエストのAPI用スキーマ
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models.quest import QuestDifficulty, QuestStatus, QuestType


class QuestBase(BaseModel):
    """クエスト基本スキーマ"""

    title: str = Field(
        ..., min_length=1, max_length=255, description="クエストタイトル"
    )
    description: str = Field(..., min_length=1, description="クエスト説明")
    quest_type: QuestType = Field(..., description="クエストタイプ")
    difficulty: QuestDifficulty = Field(..., description="クエスト難易度")
    target_skill: str = Field(
        ..., min_length=1, max_length=100, description="対象非認知能力"
    )
    estimated_duration: int = Field(..., gt=0, description="推定所要時間（分）")
    required_level: int = Field(1, ge=1, description="必要レベル")
    quest_config: Optional[Dict[str, Any]] = Field(None, description="クエスト設定")
    experience_points: int = Field(100, ge=0, description="経験値報酬")
    coins: int = Field(50, ge=0, description="コイン報酬")
    badge_id: Optional[str] = Field(None, max_length=100, description="バッジID")
    is_active: bool = Field(True, description="アクティブフラグ")
    is_daily: bool = Field(False, description="日次クエストフラグ")
    sort_order: int = Field(0, description="表示順序")


class QuestCreate(QuestBase):
    """クエスト作成スキーマ"""

    pass


class QuestUpdate(BaseModel):
    """クエスト更新スキーマ"""

    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, min_length=1)
    difficulty: Optional[QuestDifficulty] = None
    target_skill: Optional[str] = Field(None, min_length=1, max_length=100)
    estimated_duration: Optional[int] = Field(None, gt=0)
    required_level: Optional[int] = Field(None, ge=1)
    quest_config: Optional[Dict[str, Any]] = None
    experience_points: Optional[int] = Field(None, ge=0)
    coins: Optional[int] = Field(None, ge=0)
    badge_id: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None
    is_daily: Optional[bool] = None
    sort_order: Optional[int] = None


class QuestResponse(QuestBase):
    """クエスト応答スキーマ"""

    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class QuestProgressBase(BaseModel):
    """クエスト進捗基本スキーマ"""

    status: QuestStatus = Field(QuestStatus.NOT_STARTED, description="進捗ステータス")
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0, description="進捗率")
    current_step: int = Field(0, ge=0, description="現在のステップ")
    total_steps: int = Field(1, ge=1, description="総ステップ数")
    quest_data: Optional[Dict[str, Any]] = Field(None, description="クエスト固有データ")
    started_date: Optional[datetime] = Field(None, description="開始日時")
    completed_date: Optional[datetime] = Field(None, description="完了日時")
    streak_count: int = Field(0, ge=0, description="連続達成回数")
    self_evaluation: Optional[int] = Field(None, ge=1, le=5, description="自己評価")
    teacher_feedback: Optional[str] = Field(None, description="教師フィードバック")
    ai_feedback: Optional[str] = Field(None, description="AIフィードバック")


class QuestProgressCreate(QuestProgressBase):
    """クエスト進捗作成スキーマ"""

    user_id: int
    quest_id: int


class QuestProgressUpdate(BaseModel):
    """クエスト進捗更新スキーマ"""

    status: Optional[QuestStatus] = None
    progress_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    current_step: Optional[int] = Field(None, ge=0)
    total_steps: Optional[int] = Field(None, ge=1)
    quest_data: Optional[Dict[str, Any]] = None
    started_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    streak_count: Optional[int] = Field(None, ge=0)
    self_evaluation: Optional[int] = Field(None, ge=1, le=5)
    teacher_feedback: Optional[str] = None
    ai_feedback: Optional[str] = None


class QuestProgressResponse(QuestProgressBase):
    """クエスト進捗応答スキーマ"""

    id: int
    user_id: int
    quest_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class QuestProgressWithQuest(QuestProgressResponse):
    """クエスト進捗とクエスト情報を含む応答スキーマ"""

    quest: QuestResponse


class QuestSessionBase(BaseModel):
    """クエストセッション基本スキーマ"""

    session_data: Dict[str, Any] = Field(..., description="セッションデータ")
    is_active: bool = Field(True, description="アクティブフラグ")
    started_at: datetime = Field(
        default_factory=datetime.utcnow, description="開始日時"
    )
    completed_at: Optional[datetime] = Field(None, description="完了日時")


class QuestSessionCreate(QuestSessionBase):
    """クエストセッション作成スキーマ"""

    user_id: int
    quest_id: int


class QuestSessionResponse(QuestSessionBase):
    """クエストセッション応答スキーマ"""

    id: int
    user_id: int
    quest_id: int

    class Config:
        from_attributes = True


class QuestRewardBase(BaseModel):
    """クエスト報酬基本スキーマ"""

    reward_type: str = Field(..., min_length=1, max_length=50, description="報酬タイプ")
    reward_value: int = Field(..., ge=0, description="報酬値")
    reward_data: Optional[Dict[str, Any]] = Field(None, description="報酬追加データ")
    is_claimed: bool = Field(False, description="受取済みフラグ")
    claimed_at: Optional[datetime] = Field(None, description="受取日時")


class QuestRewardCreate(QuestRewardBase):
    """クエスト報酬作成スキーマ"""

    user_id: int
    quest_id: int


class QuestRewardResponse(QuestRewardBase):
    """クエスト報酬応答スキーマ"""

    id: int
    user_id: int
    quest_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class QuestListResponse(BaseModel):
    """クエスト一覧応答スキーマ"""

    quests: List[QuestResponse]
    total: int
    page: int
    size: int


class QuestProgressListResponse(BaseModel):
    """クエスト進捗一覧応答スキーマ"""

    progress: List[QuestProgressWithQuest]
    total: int
    page: int
    size: int


class QuestStatsResponse(BaseModel):
    """クエスト統計応答スキーマ"""

    total_quests: int
    completed_quests: int
    in_progress_quests: int
    total_experience: int
    total_coins: int
    streak_days: int
    favorite_quest_type: Optional[str] = None


class QuestRecommendationResponse(BaseModel):
    """クエスト推奨応答スキーマ"""

    recommended_quests: List[QuestResponse]
    reason: str
    based_on: Dict[str, Any]
