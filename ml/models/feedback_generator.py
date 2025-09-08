import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .emotion_analyzer import EmotionAnalysisResult, EmotionCategory


class FeedbackStyle(Enum):
    """フィードバックスタイルの定義"""

    ENCOURAGING = "encouraging"  # 励まし重視
    INSTRUCTIVE = "instructive"  # 解説重視
    BALANCED = "balanced"  # バランス型
    CHALLENGING = "challenging"  # チャレンジ促進
    SUPPORTIVE = "supportive"  # サポート重視


@dataclass
class FeedbackContext:
    """フィードバック生成のためのコンテキスト情報"""

    emotion_state: EmotionAnalysisResult
    learning_state: Dict[str, float]
    task_history: List[Dict[str, any]]
    preferred_style: FeedbackStyle


class FeedbackTemplate:
    """フィードバックテンプレート"""

    def __init__(self):
        self.templates = {
            EmotionCategory.MOTIVATED: {
                FeedbackStyle.ENCOURAGING: [
                    "その調子です！{achievement}は素晴らしい成果ですね。次の{next_goal}も、きっと達成できますよ！",
                    "やる気に満ちていますね！{progress}の進歩が見られます。その意欲を大切に、次のステップに進みましょう。",
                ],
                FeedbackStyle.INSTRUCTIVE: [
                    "{achievement}が達成できましたね。次は{next_goal}に挑戦してみましょう。{tip}というアプローチが効果的かもしれません。",
                    "順調な進歩を見せていますね。{progress}の理解が深まっています。次は{next_concept}について学んでいきましょう。",
                ],
            },
            EmotionCategory.FRUSTRATED: {
                FeedbackStyle.SUPPORTIVE: [
                    "難しく感じるのは当然です。{problem}は多くの人が躓くポイントです。一緒に解決していきましょう。",
                    "一歩ずつ着実に進んでいきましょう。{small_step}から始めてみませんか？",
                ],
                FeedbackStyle.BALANCED: [
                    "焦る必要はありません。{achievement}まで到達できているのは素晴らしいことです。次は{next_step}に注目してみましょう。",
                    "困難に直面するのは成長の証です。{current_point}を理解できれば、大きく前進できますよ。",
                ],
            },
            EmotionCategory.CONFUSED: {
                FeedbackStyle.INSTRUCTIVE: [
                    "整理して考えてみましょう。まず、{basic_concept}を確認します。そこから{next_point}に進んでいきます。",
                    "{problem}について混乱しているようですね。{simple_example}を例に、一緒に考えていきましょう。",
                ],
                FeedbackStyle.SUPPORTIVE: [
                    "少し複雑に感じますか？{key_point}に焦点を当てて、順番に理解していきましょう。",
                    "一つずつ整理していけば、必ず理解できます。まずは{first_step}から始めてみましょう。",
                ],
            },
            EmotionCategory.SATISFIED: {
                FeedbackStyle.CHALLENGING: [
                    "素晴らしい達成です！さらなる高みを目指して、{advanced_topic}に挑戦してみませんか？",
                    "着実な進歩を遂げていますね。次は{challenging_task}に挑戦する準備が整っていると思います。",
                ],
                FeedbackStyle.BALANCED: [
                    "良い調子ですね。{current_achievement}の理解が深まっています。この調子で{next_level}も極めていきましょう。",
                    "満足できる結果が出ていますね。この勢いを維持しつつ、{new_aspect}にも目を向けてみましょう。",
                ],
            },
            EmotionCategory.ANXIOUS: {
                FeedbackStyle.SUPPORTIVE: [
                    "心配することはありません。{progress}まで来られたのは、大きな成果です。一緒に次のステップを考えていきましょう。",
                    "不安を感じるのは自然なことです。{achievement}ができているのは、確かな証拠です。自信を持って進みましょう。",
                ],
                FeedbackStyle.ENCOURAGING: [
                    "あなたならできます！{past_success}を思い出してください。同じように{current_challenge}も乗り越えられますよ。",
                    "一歩ずつ、着実に進んでいきましょう。{small_win}から始めて、徐々に自信をつけていきましょう。",
                ],
            },
            EmotionCategory.CONFIDENT: {
                FeedbackStyle.CHALLENGING: [
                    "その自信は本物ですね！さらなる高みを目指して、{advanced_challenge}に挑戦してみませんか？",
                    "素晴らしい自信を持っていますね。その力を活かして、{complex_problem}に取り組んでみましょう。",
                ],
                FeedbackStyle.BALANCED: [
                    "着実な進歩が自信につながっていますね。次は{new_territory}の探索に挑戦してみましょう。",
                    "その自信を大切に、さらなる高みを目指しましょう。{next_challenge}が、新たな発見をもたらすはずです。",
                ],
            },
            EmotionCategory.BORED: {
                FeedbackStyle.CHALLENGING: [
                    "新しい挑戦をご用意しました！{interesting_task}に取り組んでみませんか？きっと刺激的な発見があるはずです。",
                    "退屈を感じているようですね。{creative_challenge}で、あなたの創造性を存分に発揮してみましょう。",
                ],
                FeedbackStyle.ENCOURAGING: [
                    "視点を変えて見ると、新しい発見があるかもしれません。{alternative_approach}を試してみませんか？",
                    "あなたの能力なら、もっと面白い課題に挑戦できるはずです。{advanced_project}に取り組んでみましょう。",
                ],
            },
        }


class FeedbackGenerator:
    """フィードバック生成システム"""

    def __init__(self, model_name: str = "sonoisa/t5-base-japanese-v1.1"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # T5モデルとトークナイザーの初期化
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )

        # テンプレートの初期化
        self.template = FeedbackTemplate()

        # ロガーの設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def generate_feedback(self, context: FeedbackContext) -> str:
        """フィードバックの生成"""
        # 感情状態と学習状態に基づいてフィードバックスタイルを決定
        if not context.preferred_style:
            style = self._determine_feedback_style(context)
        else:
            style = context.preferred_style

        # テンプレートの選択
        template = self._select_template(context.emotion_state.primary_emotion, style)

        # コンテキスト情報の抽出
        context_info = self._extract_context_info(context)

        # テンプレートの補完
        base_feedback = template.format(**context_info)

        # T5モデルによるフィードバックの生成
        enhanced_feedback = self._enhance_feedback(base_feedback, context)

        return enhanced_feedback

    def _determine_feedback_style(self, context: FeedbackContext) -> FeedbackStyle:
        """フィードバックスタイルの決定"""
        emotion = context.emotion_state.primary_emotion
        learning_state = context.learning_state

        # 感情状態に基づく基本スタイルの決定
        if emotion == EmotionCategory.FRUSTRATED or emotion == EmotionCategory.ANXIOUS:
            return FeedbackStyle.SUPPORTIVE
        elif emotion == EmotionCategory.CONFUSED:
            return FeedbackStyle.INSTRUCTIVE
        elif (
            emotion == EmotionCategory.MOTIVATED or emotion == EmotionCategory.CONFIDENT
        ):
            return FeedbackStyle.CHALLENGING
        elif emotion == EmotionCategory.BORED:
            return FeedbackStyle.ENCOURAGING
        else:
            return FeedbackStyle.BALANCED

    def _select_template(self, emotion: EmotionCategory, style: FeedbackStyle) -> str:
        """テンプレートの選択"""
        templates = self.template.templates[emotion][style]
        return templates[torch.randint(0, len(templates), (1,)).item()]

    def _extract_context_info(self, context: FeedbackContext) -> Dict[str, str]:
        """コンテキスト情報の抽出"""
        # 学習状態からの情報抽出
        current_level = context.learning_state.get("current_level", 0)
        success_rate = context.learning_state.get("success_rate", 0.0)
        challenge_count = context.learning_state.get("challenge_count", 0)

        # タスク履歴からの情報抽出
        recent_achievements = self._extract_recent_achievements(context.task_history)
        next_goals = self._determine_next_goals(context)

        return {
            "achievement": (
                recent_achievements[0] if recent_achievements else "これまでの努力"
            ),
            "next_goal": next_goals[0] if next_goals else "次の課題",
            "progress": f"レベル{current_level}での{success_rate*100:.0f}%の成功率",
            "problem": self._identify_current_problem(context),
            "small_step": self._suggest_small_step(context),
            "current_point": self._identify_key_learning_point(context),
            "basic_concept": self._identify_basic_concept(context),
            "next_point": self._identify_next_learning_point(context),
            "key_point": self._identify_key_point(context),
            "first_step": self._suggest_first_step(context),
            "advanced_topic": next_goals[-1] if next_goals else "より高度な課題",
            "challenging_task": self._suggest_challenging_task(context),
            "new_aspect": self._identify_new_aspect(context),
            "past_success": (
                recent_achievements[-1] if recent_achievements else "これまでの成功体験"
            ),
            "current_challenge": self._identify_current_challenge(context),
            "small_win": self._suggest_small_win(context),
            "advanced_challenge": self._suggest_advanced_challenge(context),
            "complex_problem": self._suggest_complex_problem(context),
            "new_territory": self._suggest_new_territory(context),
            "next_challenge": self._suggest_next_challenge(context),
            "interesting_task": self._suggest_interesting_task(context),
            "creative_challenge": self._suggest_creative_challenge(context),
            "alternative_approach": self._suggest_alternative_approach(context),
            "advanced_project": self._suggest_advanced_project(context),
        }

    def _enhance_feedback(self, base_feedback: str, context: FeedbackContext) -> str:
        """T5モデルによるフィードバックの強化"""
        # 入力テキストの準備
        input_text = f"フィードバック: {base_feedback}\n感情: {context.emotion_state.primary_emotion.value}"

        # トークン化
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        # フィードバックの生成
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=200,
                num_beams=5,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )

        # デコード
        enhanced_feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return enhanced_feedback

    def _extract_recent_achievements(
        self, task_history: List[Dict[str, any]]
    ) -> List[str]:
        """最近の成果の抽出"""
        achievements = []
        for task in reversed(task_history[-5:]):  # 直近5つのタスクを確認
            if task.get("success", False):
                achievements.append(task.get("description", "課題の達成"))
        return achievements

    def _determine_next_goals(self, context: FeedbackContext) -> List[str]:
        """次の目標の決定"""
        current_level = context.learning_state.get("current_level", 0)
        success_rate = context.learning_state.get("success_rate", 0.0)

        goals = []
        if success_rate > 0.8:
            goals.append(f"レベル{current_level + 1}の課題")
            goals.append("より複雑な問題")
        elif success_rate > 0.6:
            goals.append("現在の課題の完全習得")
            goals.append("応用問題")
        else:
            goals.append("基本的な概念の理解")
            goals.append("練習問題の反復")

        return goals

    # 以下、コンテキストに基づく具体的な提案を生成するヘルパーメソッド
    def _identify_current_problem(self, context: FeedbackContext) -> str:
        """現在の問題点の特定"""
        if context.emotion_state.primary_emotion == EmotionCategory.CONFUSED:
            return "概念の関連性の理解"
        elif context.emotion_state.primary_emotion == EmotionCategory.FRUSTRATED:
            return "複雑な問題の解き方"
        return "現在の課題"

    def _suggest_small_step(self, context: FeedbackContext) -> str:
        """小さなステップの提案"""
        return "基本的な例題の見直し"

    def _identify_key_learning_point(self, context: FeedbackContext) -> str:
        """重要な学習ポイントの特定"""
        return "重要な概念のつながり"

    def _identify_basic_concept(self, context: FeedbackContext) -> str:
        """基本概念の特定"""
        return "基礎となる考え方"

    def _identify_next_learning_point(self, context: FeedbackContext) -> str:
        """次の学習ポイントの特定"""
        return "応用的な考え方"

    def _identify_key_point(self, context: FeedbackContext) -> str:
        """キーポイントの特定"""
        return "最も重要な部分"

    def _suggest_first_step(self, context: FeedbackContext) -> str:
        """最初のステップの提案"""
        return "簡単な例題"

    def _suggest_challenging_task(self, context: FeedbackContext) -> str:
        """チャレンジングなタスクの提案"""
        return "発展的な問題"

    def _identify_new_aspect(self, context: FeedbackContext) -> str:
        """新しい側面の特定"""
        return "異なる視点からのアプローチ"

    def _identify_current_challenge(self, context: FeedbackContext) -> str:
        """現在のチャレンジの特定"""
        return "目の前の課題"

    def _suggest_small_win(self, context: FeedbackContext) -> str:
        """小さな成功体験の提案"""
        return "簡単な練習問題"

    def _suggest_advanced_challenge(self, context: FeedbackContext) -> str:
        """高度なチャレンジの提案"""
        return "より複雑な課題"

    def _suggest_complex_problem(self, context: FeedbackContext) -> str:
        """複雑な問題の提案"""
        return "総合的な問題"

    def _suggest_new_territory(self, context: FeedbackContext) -> str:
        """新しい領域の提案"""
        return "未知の分野"

    def _suggest_next_challenge(self, context: FeedbackContext) -> str:
        """次のチャレンジの提案"""
        return "次のレベルの課題"

    def _suggest_interesting_task(self, context: FeedbackContext) -> str:
        """興味深いタスクの提案"""
        return "創造的な問題"

    def _suggest_creative_challenge(self, context: FeedbackContext) -> str:
        """創造的なチャレンジの提案"""
        return "自由な発想が必要な課題"

    def _suggest_alternative_approach(self, context: FeedbackContext) -> str:
        """代替アプローチの提案"""
        return "異なる解決方法"

    def _suggest_advanced_project(self, context: FeedbackContext) -> str:
        """高度なプロジェクトの提案"""
        return "チャレンジングなプロジェクト"
