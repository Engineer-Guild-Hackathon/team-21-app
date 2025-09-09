from typing import Any, Dict

import torch
from transformers import BertForSequenceClassification, BertModel, BertTokenizer


class DialogueSystem:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        self.model = BertModel.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        self.classifier = BertForSequenceClassification.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking",
            num_labels=5,  # 感情カテゴリの数
        )

        # デバイスの設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.classifier.to(self.device)

        # 応答テンプレート
        self.response_templates = {
            "joy": [
                "その調子ですね！その感情を大切に、次のステップに進みましょう。",
                "素晴らしい進捗ですね。この勢いを維持していきましょう。",
                "ポジティブな姿勢が素晴らしいです。その意欲を活かして、新しい課題に挑戦してみませんか？",
            ],
            "frustration": [
                "難しく感じるのは当然です。一緒に解決策を考えていきましょう。",
                "一歩ずつ着実に進んでいけば、必ず理解できるようになります。",
                "焦る必要はありません。まずは基本的なところから見直してみましょう。",
            ],
            "concentration": [
                "集中力が高まっているようですね。この状態を活かして進めていきましょう。",
                "良い調子で取り組めていますね。必要なサポートがあれば言ってください。",
                "しっかりと課題に向き合えていますね。一緒にゴールを目指しましょう。",
            ],
            "neutral": [
                "どんなことでも気軽に相談してくださいね。",
                "一緒に最適な学習方法を見つけていきましょう。",
                "あなたのペースで進めていきましょう。",
            ],
        }

    def _analyze_emotion(self, text: str) -> Dict[str, float]:
        """テキストから感情を分析"""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.classifier(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        emotion_scores = probabilities[0].cpu().numpy()
        emotions = ["joy", "frustration", "concentration", "neutral"]
        return dict(zip(emotions, emotion_scores))

    def _get_response_template(self, emotion: str, action: Dict[str, str]) -> str:
        """感情とアクションに基づいて応答テンプレートを選択"""
        templates = self.response_templates.get(
            emotion, self.response_templates["neutral"]
        )
        # アクションに基づいてテンプレートを選択するロジックを実装
        return templates[0]  # 仮の実装

    def _generate_custom_response(self, user_message: str, base_template: str) -> str:
        """BERTを使用してカスタム応答を生成"""
        # ここでBERTモデルを使用してより自然な応答を生成
        # 現在は簡易的な実装
        return base_template

    def generate_response(
        self, user_message: str, emotion: str, action: Dict[str, str]
    ) -> Dict[str, Any]:
        """ユーザーメッセージに対する応答を生成"""
        # 感情分析
        emotion_scores = self._analyze_emotion(user_message)
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]

        # 応答テンプレートの取得
        base_response = self._get_response_template(dominant_emotion, action)

        # カスタム応答の生成
        custom_response = self._generate_custom_response(user_message, base_response)

        return {
            "text": custom_response,
            "emotion": dominant_emotion,
            "intensity": float(max(emotion_scores.values())),
        }
