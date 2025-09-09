from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy.orm import Session

from ...domain.models.behavior import BehaviorRecord
from ...domain.models.emotion import EmotionRecord
from ..data_collection.emotion_data_collector import EmotionDataCollector


class DataPipeline:
    def __init__(self):
        self.emotion_collector = EmotionDataCollector()

        # 感情カテゴリの重み付け
        self.emotion_weights = {
            "joy": 1.0,
            "frustration": -0.5,
            "concentration": 0.8,
            "neutral": 0.0,
        }

        # 行動タイプの重み付け
        self.behavior_weights = {
            "problem_solving": 1.0,
            "hint_usage": -0.2,
            "review": 0.5,
            "give_up": -1.0,
        }

    def process_user_session(
        self, user_id: int, session_data: Dict[str, Any], db: Session
    ) -> Dict[str, Any]:
        """ユーザーセッションのデータを処理"""

        # 1. 感情データの処理
        emotion_data = self._process_emotion_data(
            user_id, session_data.get("emotions", [])
        )

        # 2. 行動データの処理
        behavior_data = self._process_behavior_data(
            user_id, session_data.get("behaviors", [])
        )

        # 3. データの相関分析
        correlations = self._analyze_correlations(emotion_data, behavior_data)

        # 4. 非認知能力スコアの計算
        noncog_scores = self._calculate_noncog_scores(emotion_data, behavior_data)

        # 5. データベースに保存
        self._save_to_database(db, user_id, emotion_data, behavior_data, noncog_scores)

        return {
            "emotion_summary": emotion_data["summary"],
            "behavior_summary": behavior_data["summary"],
            "correlations": correlations,
            "noncog_scores": noncog_scores,
        }

    def _process_emotion_data(
        self, user_id: int, emotion_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """感情データの処理"""

        # 感情の時系列データを作成
        emotion_df = pd.DataFrame(emotion_records)
        emotion_df["timestamp"] = pd.to_datetime(emotion_df["timestamp"])
        emotion_df.sort_values("timestamp", inplace=True)

        # 感情の推移を分析
        emotion_trends = {}
        for emotion in self.emotion_weights.keys():
            mask = emotion_df["emotion_type"] == emotion
            emotion_trends[emotion] = {
                "count": mask.sum(),
                "average_intensity": emotion_df[mask]["intensity"].mean(),
                "trend": self._calculate_trend(emotion_df[mask]["intensity"].tolist()),
            }

        # 感情の要約統計を計算
        summary = {
            "dominant_emotion": max(
                emotion_trends.items(), key=lambda x: x[1]["count"]
            )[0],
            "emotional_stability": self._calculate_stability(emotion_df["intensity"]),
            "emotion_trends": emotion_trends,
        }

        return {"raw_data": emotion_df, "trends": emotion_trends, "summary": summary}

    def _process_behavior_data(
        self, user_id: int, behavior_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """行動データの処理"""

        # 行動の時系列データを作成
        behavior_df = pd.DataFrame(behavior_records)
        behavior_df["timestamp"] = pd.to_datetime(behavior_df["timestamp"])
        behavior_df.sort_values("timestamp", inplace=True)

        # 行動パターンの分析
        behavior_patterns = {}
        for behavior in self.behavior_weights.keys():
            mask = behavior_df["action_type"] == behavior
            behavior_patterns[behavior] = {
                "count": mask.sum(),
                "success_rate": behavior_df[mask]["success_rate"].mean(),
                "trend": self._calculate_trend(
                    behavior_df[mask]["success_rate"].tolist()
                ),
            }

        # 行動の要約統計を計算
        summary = {
            "most_frequent_action": max(
                behavior_patterns.items(), key=lambda x: x[1]["count"]
            )[0],
            "average_success_rate": behavior_df["success_rate"].mean(),
            "behavior_patterns": behavior_patterns,
        }

        return {
            "raw_data": behavior_df,
            "patterns": behavior_patterns,
            "summary": summary,
        }

    def _analyze_correlations(
        self, emotion_data: Dict[str, Any], behavior_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """感情と行動の相関を分析"""

        emotion_df = emotion_data["raw_data"]
        behavior_df = behavior_data["raw_data"]

        correlations = {}

        # 30分以内の感情と行動を関連付け
        for emotion in self.emotion_weights.keys():
            for behavior in self.behavior_weights.keys():
                emotion_times = emotion_df[emotion_df["emotion_type"] == emotion][
                    "timestamp"
                ]

                behavior_times = behavior_df[behavior_df["action_type"] == behavior][
                    "timestamp"
                ]

                # 時間的な近接性に基づく相関を計算
                correlation = self._calculate_temporal_correlation(
                    emotion_times, behavior_times, window=timedelta(minutes=30)
                )

                correlations[f"{emotion}_{behavior}"] = correlation

        return correlations

    def _calculate_noncog_scores(
        self, emotion_data: Dict[str, Any], behavior_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """非認知能力スコアを計算"""

        # 感情の安定性からレジリエンススコアを計算
        resilience_score = emotion_data["summary"]["emotional_stability"]

        # 行動パターンから粘り強さスコアを計算
        persistence_score = self._calculate_persistence_score(behavior_data["patterns"])

        # 感情と行動の組み合わせから協調性スコアを計算
        cooperation_score = self._calculate_cooperation_score(
            emotion_data["summary"], behavior_data["summary"]
        )

        return {
            "resilience": resilience_score,
            "persistence": persistence_score,
            "cooperation": cooperation_score,
            "total": (resilience_score + persistence_score + cooperation_score) / 3,
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """値の傾向を計算"""
        if len(values) < 2:
            return "stable"

        # 線形回帰で傾きを計算
        x = list(range(len(values)))
        y = values

        if len(x) != len(y):
            return "stable"

        slope = pd.Series(y).diff().mean()

        if abs(slope) < 0.1:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def _calculate_stability(self, values: pd.Series) -> float:
        """値の安定性を計算"""
        if values.empty:
            return 0.0

        # 標準偏差の逆数を正規化
        std = values.std()
        if std == 0:
            return 1.0

        stability = 1 / (1 + std)
        return stability

    def _calculate_temporal_correlation(
        self, times1: pd.Series, times2: pd.Series, window: timedelta
    ) -> float:
        """時間的な相関を計算"""
        if times1.empty or times2.empty:
            return 0.0

        # 時間窓内のイベントペアをカウント
        total_pairs = 0
        matched_pairs = 0

        for t1 in times1:
            for t2 in times2:
                total_pairs += 1
                if abs(t1 - t2) <= window:
                    matched_pairs += 1

        return matched_pairs / total_pairs if total_pairs > 0 else 0.0

    def _calculate_persistence_score(
        self, behavior_patterns: Dict[str, Dict[str, Any]]
    ) -> float:
        """粘り強さスコアを計算"""

        # 諦めずに続ける傾向を評価
        give_up_rate = behavior_patterns.get("give_up", {}).get("count", 0)
        problem_solving_rate = behavior_patterns.get("problem_solving", {}).get(
            "count", 0
        )

        if problem_solving_rate == 0:
            return 0.0

        persistence = 1 - (give_up_rate / (give_up_rate + problem_solving_rate))
        return persistence

    def _calculate_cooperation_score(
        self, emotion_summary: Dict[str, Any], behavior_summary: Dict[str, Any]
    ) -> float:
        """協調性スコアを計算"""

        # 感情の安定性と建設的な行動から協調性を評価
        emotional_stability = emotion_summary.get("emotional_stability", 0.0)
        success_rate = behavior_summary.get("average_success_rate", 0.0)

        cooperation = (emotional_stability + success_rate) / 2
        return cooperation

    def _save_to_database(
        self,
        db: Session,
        user_id: int,
        emotion_data: Dict[str, Any],
        behavior_data: Dict[str, Any],
        noncog_scores: Dict[str, float],
    ):
        """処理したデータをデータベースに保存"""

        # 感情レコードの保存
        for _, row in emotion_data["raw_data"].iterrows():
            emotion_record = EmotionRecord(
                user_id=user_id,
                emotion_type=row["emotion_type"],
                intensity=row["intensity"],
                context=row.get("context", "general"),
            )
            db.add(emotion_record)

        # 行動レコードの保存
        for _, row in behavior_data["raw_data"].iterrows():
            behavior_record = BehaviorRecord(
                user_id=user_id,
                action_type=row["action_type"],
                success_rate=row["success_rate"],
                context=row.get("context", "general"),
            )
            db.add(behavior_record)

        # 変更をコミット
        db.commit()
