"""
感情分析モジュール
プレイヤーの感情状態を分析し、適切なフィードバックを生成する
"""

from typing import Dict, List, Optional

import torch
from transformers import pipeline


class EmotionAnalyzer:
    def __init__(self):
        self.text_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
        )

    def analyze_text(self, text: str) -> Dict[str, float]:
        """テキストから感情を分析する"""
        results = self.text_analyzer(text)[0]
        return {item["label"]: item["score"] for item in results}

    def get_emotional_state(
        self,
        text: Optional[str] = None,
        facial_expression: Optional[Dict] = None,
        behavior_data: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        複数の入力から総合的な感情状態を分析する

        Args:
            text: ユーザーの入力テキスト
            facial_expression: 表情分析データ
            behavior_data: 行動データ（クリック、タイピング速度など）

        Returns:
            Dict[str, float]: 各感情の強度を示す辞書
        """
        emotional_state = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "frustration": 0.0,
            "concentration": 0.0,
        }

        # テキスト分析
        if text:
            text_emotions = self.analyze_text(text)
            for emotion, score in text_emotions.items():
                if emotion in emotional_state:
                    emotional_state[emotion] += score * 0.4  # テキストの重み

        # 表情分析の統合
        if facial_expression:
            for emotion, score in facial_expression.items():
                if emotion in emotional_state:
                    emotional_state[emotion] += score * 0.4  # 表情の重み

        # 行動データの統合
        if behavior_data:
            # 行動データからの感情推定（例：素早いクリックは焦りを示すかもしれない）
            if "click_speed" in behavior_data:
                if behavior_data["click_speed"] > 2.0:  # 閾値
                    emotional_state["frustration"] += 0.2

            # 集中度の推定
            if "focus_time" in behavior_data:
                emotional_state["concentration"] = min(
                    1.0, behavior_data["focus_time"] / 300.0  # 5分を最大値とする
                )

        # 正規化（0-1の範囲に収める）
        total = sum(emotional_state.values())
        if total > 0:
            emotional_state = {k: v / total for k, v in emotional_state.items()}

        return emotional_state

    def get_feedback_suggestion(self, emotional_state: Dict[str, float]) -> str:
        """
        感情状態に基づいて適切なフィードバックを提案する

        Args:
            emotional_state: 感情状態の辞書

        Returns:
            str: フィードバックの提案
        """
        # 最も強い感情を特定
        dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])

        feedback_templates = {
            "joy": "その調子です！さらに高度な課題に挑戦してみましょう。",
            "sadness": "一休みしませんか？リフレッシュすると新しい視点が見つかるかもしれません。",
            "anger": "深呼吸をして、一つずつ問題を整理していきましょう。",
            "fear": "ゆっくり進めていきましょう。一歩一歩、確実に前に進んでいきます。",
            "surprise": "新しい発見がありましたね！その興味を大切に探求を続けましょう。",
            "frustration": "難しく感じるのは当然です。別のアプローチを試してみませんか？",
            "concentration": "素晴らしい集中力です。そのペースを保ちながら進めていきましょう。",
        }

        return feedback_templates.get(
            dominant_emotion[0], "一緒に頑張っていきましょう！"
        )
