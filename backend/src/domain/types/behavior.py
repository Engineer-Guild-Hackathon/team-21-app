from dataclasses import dataclass
from datetime import datetime
from typing import List, NewType, Optional

BehaviorId = NewType("BehaviorId", int)
UserId = NewType("UserId", int)


@dataclass(frozen=True)
class BehaviorCategory:
    """行動カテゴリ

    ユーザーの行動を分類するためのカテゴリ定義。
    例: "exercise", "sleep", "work", "social" など
    """

    name: str
    description: str
    impact_level: int  # 1-5のスケールで行動の重要度を表す


@dataclass(frozen=True)
class BehaviorMetrics:
    """行動メトリクス

    行動の定量的な測定値を表す値オブジェクト。
    例: 運動時間、睡眠時間、作業効率など
    """

    metric_name: str
    value: float
    unit: str
    timestamp: datetime


@dataclass(frozen=True)
class BehaviorPattern:
    """行動パターン

    一定期間の行動の繰り返しパターンを表す値オブジェクト。
    """

    category: BehaviorCategory
    frequency: int  # 期間内での発生回数
    average_duration: float  # 平均継続時間（分）
    time_of_day: List[str]  # ["morning", "afternoon", "evening", "night"]
    days_of_week: List[str]  # ["monday", "tuesday", ...]
    consistency_score: float  # 0-1のスケールでパターンの一貫性を表す


@dataclass(frozen=True)
class BehaviorGoal:
    """行動目標

    ユーザーが設定した行動に関する目標を表す値オブジェクト。
    """

    category: BehaviorCategory
    target_value: float
    target_unit: str
    frequency_target: str  # "daily", "weekly", "monthly"
    start_date: datetime
    end_date: Optional[datetime]
    progress: float  # 0-1のスケールで進捗を表す
    status: str  # "not_started", "in_progress", "completed", "failed"


@dataclass(frozen=True)
class BehaviorTrigger:
    """行動トリガー

    特定の行動を引き起こす要因を表す値オブジェクト。
    """

    trigger_type: str  # "emotion", "time", "location", "social"
    condition: dict  # トリガー条件（型はトリガータイプによって異なる）
    confidence: float  # 0-1のスケールでトリガーの確実性を表す
    last_triggered: Optional[datetime]
    frequency: int  # このトリガーが発生した回数
