"""
会話履歴から非認知能力を分析するMLモデル
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


@dataclass
class NonCognitiveSkills:
    """非認知能力スコア"""

    grit: float  # やり抜く力 (0.0-5.0)
    collaboration: float  # 協調性 (0.0-5.0)
    self_regulation: float  # 自己制御 (0.0-5.0)
    emotional_intelligence: float  # 感情知能 (0.0-5.0)
    confidence: float = 0.0  # 自信 (0.0-5.0)


@dataclass
class ConversationMetrics:
    """会話のメトリクス"""

    message_count: int
    avg_message_length: float
    question_count: int
    positive_words_count: int
    negative_words_count: int
    learning_keywords_count: int
    help_seeking_count: int
    completion_rate: float  # 問題解決完了率


class ConversationAnalyzer:
    """会話履歴から非認知能力を分析するクラス"""

    def __init__(self):
        self.grit_model = None
        self.collaboration_model = None
        self.self_regulation_model = None
        self.emotional_intelligence_model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self.scaler = StandardScaler()
        self._load_models()

    def _load_models(self):
        """事前訓練済みモデルを読み込む"""
        models_dir = os.path.join(os.path.dirname(__file__), "trained_models")

        if os.path.exists(models_dir):
            try:
                self.grit_model = joblib.load(
                    os.path.join(models_dir, "grit_model.pkl")
                )
                self.collaboration_model = joblib.load(
                    os.path.join(models_dir, "collaboration_model.pkl")
                )
                self.self_regulation_model = joblib.load(
                    os.path.join(models_dir, "self_regulation_model.pkl")
                )
                self.emotional_intelligence_model = joblib.load(
                    os.path.join(models_dir, "emotional_intelligence_model.pkl")
                )
                self.vectorizer = joblib.load(
                    os.path.join(models_dir, "vectorizer.pkl")
                )
                self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
            except FileNotFoundError:
                print(
                    "事前訓練済みモデルが見つかりません。デフォルトモデルを使用します。"
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

    def _create_rule_based_model(self):
        """ルールベースのダミーモデル"""

        class RuleBasedModel:
            def predict(self, X):
                return np.random.uniform(1.0, 4.0, len(X))

        return RuleBasedModel()

    def analyze_conversation(self, messages: List[Dict]) -> NonCognitiveSkills:
        """会話履歴を分析して非認知能力スコアを算出"""

        # 会話メトリクスを計算
        metrics = self._calculate_conversation_metrics(messages)

        # テキストデータを準備
        conversation_text = self._extract_conversation_text(messages)

        # 特徴量を抽出
        features = self._extract_features(conversation_text, metrics)

        # 各非認知能力スコアを予測
        grit_score = self._predict_grit(features)
        collaboration_score = self._predict_collaboration(features)
        self_regulation_score = self._predict_self_regulation(features)
        emotional_intelligence_score = self._predict_emotional_intelligence(features)
        confidence_score = self._predict_confidence(features)

        return NonCognitiveSkills(
            grit=grit_score,
            collaboration=collaboration_score,
            self_regulation=self_regulation_score,
            emotional_intelligence=emotional_intelligence_score,
            confidence=confidence_score,
        )

    def _calculate_conversation_metrics(
        self, messages: List[Dict]
    ) -> ConversationMetrics:
        """会話のメトリクスを計算"""
        user_messages = [msg for msg in messages if msg.get("role") == "user"]

        total_messages = len(user_messages)
        total_length = sum(len(msg.get("content", "")) for msg in user_messages)
        avg_length = total_length / max(total_messages, 1)

        # 質問の数をカウント
        question_count = sum(
            1
            for msg in user_messages
            if "?" in msg.get("content", "")
            or "ですか" in msg.get("content", "")
            or "でしょうか" in msg.get("content", "")
        )

        # ポジティブ・ネガティブワードをカウント
        positive_words = [
            "頑張る",
            "努力",
            "続ける",
            "挑戦",
            "学ぶ",
            "理解",
            "できた",
            "成功",
            "嬉しい",
            "楽しい",
        ]
        negative_words = [
            "難しい",
            "わからない",
            "できない",
            "困った",
            "疲れた",
            "嫌い",
            "苦手",
            "失敗",
        ]

        all_text = " ".join(msg.get("content", "") for msg in user_messages)
        positive_count = sum(all_text.count(word) for word in positive_words)
        negative_count = sum(all_text.count(word) for word in negative_words)

        # 学習関連キーワードをカウント
        learning_keywords = [
            "勉強",
            "学習",
            "問題",
            "課題",
            "宿題",
            "授業",
            "教科書",
            "テスト",
            "試験",
        ]
        learning_keywords_count = sum(
            all_text.count(word) for word in learning_keywords
        )

        # ヘルプシーキングをカウント
        help_words = [
            "教えて",
            "助けて",
            "手伝って",
            "わからない",
            "困っている",
            "どうすれば",
        ]
        help_seeking_count = sum(all_text.count(word) for word in help_words)

        # 完了率（簡易版：質問に対する返答の有無）
        completion_rate = min(
            1.0, total_messages / max(1, len(messages) - total_messages)
        )

        return ConversationMetrics(
            message_count=total_messages,
            avg_message_length=avg_length,
            question_count=question_count,
            positive_words_count=positive_count,
            negative_words_count=negative_count,
            learning_keywords_count=learning_keywords_count,
            help_seeking_count=help_seeking_count,
            completion_rate=completion_rate,
        )

    def _extract_conversation_text(self, messages: List[Dict]) -> str:
        """会話からテキストを抽出"""
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        return " ".join(msg.get("content", "") for msg in user_messages)

    def _extract_features(
        self, conversation_text: str, metrics: ConversationMetrics
    ) -> np.ndarray:
        """特徴量を抽出"""
        features = [
            metrics.message_count,
            metrics.avg_message_length,
            metrics.question_count,
            metrics.positive_words_count,
            metrics.negative_words_count,
            metrics.learning_keywords_count,
            metrics.help_seeking_count,
            metrics.completion_rate,
            len(conversation_text.split()),
            len(re.findall(r"[!！]", conversation_text)),  # 感嘆符の数
            len(re.findall(r"[?？]", conversation_text)),  # 疑問符の数
        ]

        return np.array(features).reshape(1, -1)

    def _predict_grit(self, features: np.ndarray) -> float:
        """グリット（やり抜く力）を予測"""
        if self.grit_model:
            score = self.grit_model.predict(features)[0]
        else:
            # ルールベースの簡易予測
            score = min(
                5.0, max(1.0, 2.0 + features[0][2] * 0.1 + features[0][3] * 0.2)
            )
        return round(score, 2)

    def _predict_collaboration(self, features: np.ndarray) -> float:
        """協調性を予測"""
        if self.collaboration_model:
            score = self.collaboration_model.predict(features)[0]
        else:
            # ルールベースの簡易予測
            score = min(
                5.0, max(1.0, 2.0 + features[0][6] * 0.3 + features[0][9] * 0.1)
            )
        return round(score, 2)

    def _predict_self_regulation(self, features: np.ndarray) -> float:
        """自己制御を予測"""
        if self.self_regulation_model:
            score = self.self_regulation_model.predict(features)[0]
        else:
            # ルールベースの簡易予測
            score = min(
                5.0, max(1.0, 2.0 + features[0][1] * 0.1 + features[0][7] * 0.4)
            )
        return round(score, 2)

    def _predict_emotional_intelligence(self, features: np.ndarray) -> float:
        """感情知能を予測"""
        if self.emotional_intelligence_model:
            score = self.emotional_intelligence_model.predict(features)[0]
        else:
            # ルールベースの簡易予測
            score = min(
                5.0, max(1.0, 2.0 + features[0][3] * 0.2 - features[0][4] * 0.1)
            )
        return round(score, 2)

    def _predict_confidence(self, features: np.ndarray) -> float:
        """自信を予測"""
        # ルールベースの簡易予測
        score = min(5.0, max(1.0, 2.0 + features[0][3] * 0.2 + features[0][7] * 0.3))
        return round(score, 2)

    def generate_feedback(
        self,
        skills: NonCognitiveSkills,
        previous_skills: Optional[NonCognitiveSkills] = None,
    ) -> str:
        """非認知能力スコアに基づいてフィードバックを生成"""

        feedback_parts = []

        # 各スキルのフィードバック
        if skills.grit >= 4.0:
            feedback_parts.append(
                "🌟 素晴らしいやり抜く力を持っています！困難な課題にも諦めずに取り組む姿勢が見られます。"
            )
        elif skills.grit >= 3.0:
            feedback_parts.append(
                "👍 やり抜く力が向上しています。目標を設定して継続的に取り組んでみましょう。"
            )
        else:
            feedback_parts.append(
                "💪 やり抜く力を鍛えるために、小さな目標から始めて達成感を積み重ねていきましょう。"
            )

        if skills.collaboration >= 4.0:
            feedback_parts.append(
                "🤝 協調性がとても高いです！他者との協力を大切にしていますね。"
            )
        elif skills.collaboration >= 3.0:
            feedback_parts.append(
                "👥 協調性が育っています。グループ学習やペア学習を活用してみましょう。"
            )
        else:
            feedback_parts.append(
                "🤝 協調性を高めるために、友達と一緒に勉強したり、質問を積極的にしてみましょう。"
            )

        if skills.self_regulation >= 4.0:
            feedback_parts.append(
                "🎯 自己制御力が優れています！計画的に学習を進められています。"
            )
        elif skills.self_regulation >= 3.0:
            feedback_parts.append(
                "📝 自己制御力が向上しています。学習計画を立てて実行してみましょう。"
            )
        else:
            feedback_parts.append(
                "⏰ 自己制御力を高めるために、学習時間を決めて集中して取り組んでみましょう。"
            )

        if skills.emotional_intelligence >= 4.0:
            feedback_parts.append(
                "💝 感情知能が高いです！自分の感情を理解し、適切に表現できています。"
            )
        elif skills.emotional_intelligence >= 3.0:
            feedback_parts.append(
                "😊 感情知能が育っています。感情を言葉で表現する練習をしてみましょう。"
            )
        else:
            feedback_parts.append(
                "💭 感情知能を高めるために、自分の気持ちを振り返る時間を作ってみましょう。"
            )

        # 進歩の評価
        if previous_skills:
            improvements = []
            if skills.grit > previous_skills.grit + 0.2:
                improvements.append("やり抜く力")
            if skills.collaboration > previous_skills.collaboration + 0.2:
                improvements.append("協調性")
            if skills.self_regulation > previous_skills.self_regulation + 0.2:
                improvements.append("自己制御力")
            if (
                skills.emotional_intelligence
                > previous_skills.emotional_intelligence + 0.2
            ):
                improvements.append("感情知能")

            if improvements:
                feedback_parts.append(
                    f"🎉 素晴らしい進歩です！{', '.join(improvements)}が向上しています。"
                )

        return "\n\n".join(feedback_parts)
