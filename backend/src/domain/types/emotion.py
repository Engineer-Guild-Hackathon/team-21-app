from dataclasses import dataclass
from datetime import datetime
from typing import List, NewType

EmotionId = NewType("EmotionId", int)
UserId = NewType("UserId", int)


@dataclass(frozen=True)
class EmotionScore:
    """感情スコア

    各感情カテゴリのスコアを表す値オブジェクト。
    スコアは0-1の範囲で正規化されています。
    """

    category: str
    score: float
    confidence: float

    def __post_init__(self):
        if not 0 <= self.score <= 1:
            raise ValueError("Score must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass(frozen=True)
class EmotionAnalysis:
    """感情分析結果

    テキストまたは画像から検出された感情の分析結果。
    複数の感情カテゴリのスコアを含みます。
    """

    scores: List[EmotionScore]
    dominant_emotion: str
    timestamp: datetime
    source_type: str  # "text" or "image"
    source_content: str


@dataclass(frozen=True)
class EmotionTrend:
    """感情トレンド

    一定期間の感情の変化を表す値オブジェクト。
    """

    start_time: datetime
    end_time: datetime
    emotion_changes: List[EmotionScore]
    trend_direction: str  # "improving", "declining", "stable"
    confidence: float
