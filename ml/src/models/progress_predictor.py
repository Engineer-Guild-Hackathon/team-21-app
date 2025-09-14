"""
学習行動から進捗を予測するMLモデル
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class LearningActivity:
    """学習活動データ"""

    timestamp: datetime
    activity_type: str  # 'chat', 'quest', 'avatar_change', 'login'
    duration_minutes: int
    success_rate: float  # 0.0-1.0
    difficulty_level: float  # 1.0-5.0
    engagement_score: float  # 0.0-1.0


@dataclass
class ProgressPrediction:
    """進捗予測結果"""

    grit_improvement: float
    collaboration_improvement: float
    self_regulation_improvement: float
    emotional_intelligence_improvement: float
    confidence_improvement: float
    overall_progress: float
    recommended_activities: List[str]
    predicted_completion_time: Optional[datetime] = None


class ProgressPredictor:
    """学習行動から進捗を予測するクラス"""

    def __init__(self):
        self.grit_model = None
        self.collaboration_model = None
        self.self_regulation_model = None
        self.emotional_intelligence_model = None
        self.confidence_model = None
        self.scaler = StandardScaler()
        self._load_models()

    def _load_models(self):
        """事前訓練済みモデルを読み込む"""
        models_dir = os.path.join(os.path.dirname(__file__), "trained_models")

        if os.path.exists(models_dir):
            try:
                self.grit_model = joblib.load(
                    os.path.join(models_dir, "grit_progress_model.pkl")
                )
                self.collaboration_model = joblib.load(
                    os.path.join(models_dir, "collaboration_progress_model.pkl")
                )
                self.self_regulation_model = joblib.load(
                    os.path.join(models_dir, "self_regulation_progress_model.pkl")
                )
                self.emotional_intelligence_model = joblib.load(
                    os.path.join(
                        models_dir, "emotional_intelligence_progress_model.pkl"
                    )
                )
                self.confidence_model = joblib.load(
                    os.path.join(models_dir, "confidence_progress_model.pkl")
                )
                self.scaler = joblib.load(
                    os.path.join(models_dir, "progress_scaler.pkl")
                )
            except FileNotFoundError:
                print(
                    "事前訓練済み進捗モデルが見つかりません。デフォルトモデルを使用します。"
                )
                self._create_default_models()
        else:
            self._create_default_models()

    def _create_default_models(self):
        """デフォルトのルールベースモデルを作成"""
        self.grit_model = self._create_rule_based_model()
        self.collaboration_model = self._create_rule_based_model()
        self.self_regulation_model = self._create_rule_based_model()
        self.emotional_intelligence_model = self._create_rule_based_model()
        self.confidence_model = self._create_rule_based_model()

    def _create_rule_based_model(self):
        """ルールベースのダミーモデル"""

        class RuleBasedModel:
            def predict(self, X):
                return np.random.uniform(0.0, 0.5, len(X))

        return RuleBasedModel()

    def predict_progress(
        self,
        activities: List[LearningActivity],
        current_skills: Dict[str, float],
        time_horizon_days: int = 7,
    ) -> ProgressPrediction:
        """学習活動から進捗を予測"""

        # 活動データから特徴量を抽出
        features = self._extract_activity_features(
            activities, current_skills, time_horizon_days
        )

        # 各スキルの改善度を予測
        grit_improvement = self._predict_grit_improvement(features)
        collaboration_improvement = self._predict_collaboration_improvement(features)
        self_regulation_improvement = self._predict_self_regulation_improvement(
            features
        )
        emotional_intelligence_improvement = (
            self._predict_emotional_intelligence_improvement(features)
        )
        confidence_improvement = self._predict_confidence_improvement(features)

        # 総合進捗を計算
        overall_progress = (
            grit_improvement
            + collaboration_improvement
            + self_regulation_improvement
            + emotional_intelligence_improvement
            + confidence_improvement
        ) / 5

        # 推奨活動を生成
        recommended_activities = self._generate_recommendations(
            current_skills,
            grit_improvement,
            collaboration_improvement,
            self_regulation_improvement,
            emotional_intelligence_improvement,
            confidence_improvement,
        )

        # 完了予測時間を計算
        predicted_completion_time = self._predict_completion_time(
            current_skills, overall_progress, time_horizon_days
        )

        return ProgressPrediction(
            grit_improvement=grit_improvement,
            collaboration_improvement=collaboration_improvement,
            self_regulation_improvement=self_regulation_improvement,
            emotional_intelligence_improvement=emotional_intelligence_improvement,
            confidence_improvement=confidence_improvement,
            overall_progress=overall_progress,
            recommended_activities=recommended_activities,
            predicted_completion_time=predicted_completion_time,
        )

    def _extract_activity_features(
        self,
        activities: List[LearningActivity],
        current_skills: Dict[str, float],
        time_horizon_days: int,
    ) -> np.ndarray:
        """活動データから特徴量を抽出"""

        if not activities:
            return np.zeros(20)  # デフォルト特徴量

        # 時間範囲内の活動をフィルタ
        cutoff_date = datetime.now() - timedelta(days=time_horizon_days)
        recent_activities = [a for a in activities if a.timestamp >= cutoff_date]

        # 基本統計
        total_duration = sum(a.duration_minutes for a in recent_activities)
        avg_duration = total_duration / max(len(recent_activities), 1)
        total_sessions = len(recent_activities)

        # 活動タイプ別統計
        chat_activities = [a for a in recent_activities if a.activity_type == "chat"]
        quest_activities = [a for a in recent_activities if a.activity_type == "quest"]

        chat_duration = sum(a.duration_minutes for a in chat_activities)
        quest_duration = sum(a.duration_minutes for a in quest_activities)

        # 成功率とエンゲージメント
        avg_success_rate = np.mean([a.success_rate for a in recent_activities])
        avg_engagement = np.mean([a.engagement_score for a in recent_activities])
        avg_difficulty = np.mean([a.difficulty_level for a in recent_activities])

        # 活動の多様性
        activity_diversity = len(set(a.activity_type for a in recent_activities))

        # 連続性（最近の活動の頻度）
        if len(recent_activities) > 1:
            recent_activities.sort(key=lambda x: x.timestamp)
            time_gaps = [
                (
                    recent_activities[i + 1].timestamp - recent_activities[i].timestamp
                ).total_seconds()
                / 3600
                for i in range(len(recent_activities) - 1)
            ]
            avg_time_gap = np.mean(time_gaps) if time_gaps else 24
        else:
            avg_time_gap = 24

        # 現在のスキルレベル
        current_grit = current_skills.get("grit", 2.0)
        current_collaboration = current_skills.get("collaboration", 2.0)
        current_self_regulation = current_skills.get("self_regulation", 2.0)
        current_emotional_intelligence = current_skills.get(
            "emotional_intelligence", 2.0
        )
        current_confidence = current_skills.get("confidence", 2.0)

        # 特徴量ベクトル
        features = np.array(
            [
                total_duration,
                avg_duration,
                total_sessions,
                chat_duration,
                quest_duration,
                avg_success_rate,
                avg_engagement,
                avg_difficulty,
                activity_diversity,
                avg_time_gap,
                current_grit,
                current_collaboration,
                current_self_regulation,
                current_emotional_intelligence,
                current_confidence,
                len(chat_activities),
                len(quest_activities),
                max([a.engagement_score for a in recent_activities], default=0),
                min([a.success_rate for a in recent_activities], default=0),
                time_horizon_days,
            ]
        )

        return features.reshape(1, -1)

    def _predict_grit_improvement(self, features: np.ndarray) -> float:
        """グリット改善度を予測"""
        if self.grit_model:
            improvement = self.grit_model.predict(features)[0]
        else:
            # ルールベース予測
            duration = features[0][0]  # 総学習時間
            engagement = features[0][6]  # 平均エンゲージメント
            difficulty = features[0][7]  # 平均難易度
            improvement = min(
                1.0, max(0.0, (duration * 0.001 + engagement * 0.3 + difficulty * 0.1))
            )
        return round(improvement, 3)

    def _predict_collaboration_improvement(self, features: np.ndarray) -> float:
        """協調性改善度を予測"""
        if self.collaboration_model:
            improvement = self.collaboration_model.predict(features)[0]
        else:
            # ルールベース予測
            chat_duration = features[0][3]  # チャット時間
            sessions = features[0][2]  # セッション数
            improvement = min(1.0, max(0.0, (chat_duration * 0.0005 + sessions * 0.05)))
        return round(improvement, 3)

    def _predict_self_regulation_improvement(self, features: np.ndarray) -> float:
        """自己制御改善度を予測"""
        if self.self_regulation_model:
            improvement = self.self_regulation_model.predict(features)[0]
        else:
            # ルールベース予測
            consistency = 1.0 / max(features[0][9], 1)  # 時間ギャップの逆数
            success_rate = features[0][5]  # 成功率
            improvement = min(1.0, max(0.0, consistency * 0.4 + success_rate * 0.3))
        return round(improvement, 3)

    def _predict_emotional_intelligence_improvement(
        self, features: np.ndarray
    ) -> float:
        """感情知能改善度を予測"""
        if self.emotional_intelligence_model:
            improvement = self.emotional_intelligence_model.predict(features)[0]
        else:
            # ルールベース予測
            engagement = features[0][6]  # エンゲージメント
            chat_sessions = features[0][15]  # チャットセッション数
            improvement = min(1.0, max(0.0, engagement * 0.4 + chat_sessions * 0.02))
        return round(improvement, 3)

    def _predict_confidence_improvement(self, features: np.ndarray) -> float:
        """自信改善度を予測"""
        if self.confidence_model:
            improvement = self.confidence_model.predict(features)[0]
        else:
            # ルールベース予測
            success_rate = features[0][5]  # 成功率
            difficulty = features[0][7]  # 難易度
            improvement = min(1.0, max(0.0, success_rate * 0.5 + difficulty * 0.1))
        return round(improvement, 3)

    def _generate_recommendations(
        self,
        current_skills: Dict[str, float],
        grit_imp: float,
        collab_imp: float,
        self_reg_imp: float,
        emotional_imp: float,
        confidence_imp: float,
    ) -> List[str]:
        """推奨活動を生成"""

        recommendations = []

        # 最も改善が必要なスキルを特定
        skill_improvements = {
            "grit": grit_imp,
            "collaboration": collab_imp,
            "self_regulation": self_reg_imp,
            "emotional_intelligence": emotional_imp,
            "confidence": confidence_imp,
        }

        # 最も改善度が低いスキルを優先
        min_skill = min(skill_improvements, key=skill_improvements.get)

        if min_skill == "grit" or current_skills.get("grit", 0) < 3.0:
            recommendations.extend(
                [
                    "🎯 目標設定クエストに挑戦して、やり抜く力を鍛えましょう",
                    "📚 難しい問題に段階的に取り組んでみましょう",
                    "🏆 小さな達成を積み重ねて自信をつけましょう",
                ]
            )

        if min_skill == "collaboration" or current_skills.get("collaboration", 0) < 3.0:
            recommendations.extend(
                [
                    "🤝 AIチャットで積極的に質問してみましょう",
                    "👥 グループ学習のクエストに参加しましょう",
                    "💬 学習の悩みを共有してみましょう",
                ]
            )

        if (
            min_skill == "self_regulation"
            or current_skills.get("self_regulation", 0) < 3.0
        ):
            recommendations.extend(
                [
                    "⏰ 学習時間を決めて計画的に取り組みましょう",
                    "📝 学習記録をつけて振り返りをしましょう",
                    "🎯 優先順位をつけて効率的に学習しましょう",
                ]
            )

        if (
            min_skill == "emotional_intelligence"
            or current_skills.get("emotional_intelligence", 0) < 3.0
        ):
            recommendations.extend(
                [
                    "💭 自分の感情を振り返る時間を作りましょう",
                    "😊 ポジティブな学習体験を積み重ねましょう",
                    "🤗 困った時はAIに相談してみましょう",
                ]
            )

        if min_skill == "confidence" or current_skills.get("confidence", 0) < 3.0:
            recommendations.extend(
                [
                    "✨ 得意な分野から始めて成功体験を積みましょう",
                    "🌟 自分の成長を認めて自信をつけましょう",
                    "🎉 小さな達成も大切にしましょう",
                ]
            )

        return recommendations[:5]  # 最大5つまで

    def _predict_completion_time(
        self,
        current_skills: Dict[str, float],
        overall_progress: float,
        time_horizon_days: int,
    ) -> Optional[datetime]:
        """目標達成予測時間を計算"""

        # 目標スキルレベル（例：全スキル3.5以上）
        target_level = 3.5
        current_avg = sum(current_skills.values()) / len(current_skills)

        if current_avg >= target_level:
            return datetime.now()  # 既に達成

        # 改善に必要な時間を推定
        remaining_improvement = target_level - current_avg
        estimated_days = (
            remaining_improvement / max(overall_progress, 0.001) * time_horizon_days
        )

        return datetime.now() + timedelta(days=int(estimated_days))
