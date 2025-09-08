"""
BERTを使用した対話システム
プレイヤーとの自然な対話を生成し、非認知能力の評価も行う
"""
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

class DialogueSystem:
    def __init__(self):
        """対話システムの初期化"""
        # 感情分析用のモデル
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        
        # 非認知能力評価用のモデル
        self.noncog_classifier = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=5  # 評価する非認知能力の数
        )
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # 応答生成用の対話テンプレート
        self.dialogue_templates = {
            'frustration': [
                "大丈夫ですよ。一緒に考えていきましょう。",
                "難しく感じるのは当然です。どの部分が特に難しいですか？",
                "別のアプローチを試してみませんか？",
            ],
            'joy': [
                "その調子です！とても良い進み方ですね。",
                "新しい課題に挑戦する準備はできていますか？",
                "その経験を次の課題にも活かしていけますよ。",
            ],
            'concentration': [
                "集中力が素晴らしいですね。",
                "良いペースで進められていますね。",
                "必要なサポートがあれば言ってくださいね。",
            ],
            'confusion': [
                "少し整理してみましょうか？",
                "どの部分が分かりにくいですか？",
                "一つずつ確認していきましょう。",
            ]
        }
        
    def evaluate_noncognitive_skills(self, dialogue_history: List[str]) -> Dict[str, float]:
        """
        対話履歴から非認知能力を評価
        
        Args:
            dialogue_history: ユーザーとの対話履歴
            
        Returns:
            Dict[str, float]: 各非認知能力のスコア
        """
        # 対話履歴を結合
        combined_text = " ".join(dialogue_history[-5:])  # 直近5つの対話を使用
        
        # トークン化
        inputs = self.tokenizer(
            combined_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # 非認知能力の評価
        with torch.no_grad():
            outputs = self.noncog_classifier(**inputs)
            scores = torch.sigmoid(outputs.logits).squeeze().numpy()
        
        # スコアを辞書形式で返す
        skills = [
            "perseverance",      # やり抜く力
            "cooperation",       # 協調性
            "self_control",     # 自己制御
            "curiosity",        # 好奇心
            "problem_solving"   # 問題解決能力
        ]
        
        return dict(zip(skills, scores))
    
    def generate_response(self,
                        user_input: str,
                        emotional_state: Dict[str, float],
                        dialogue_history: List[str]) -> str:
        """
        ユーザーの入力と感情状態に基づいて応答を生成
        
        Args:
            user_input: ユーザーの入力テキスト
            emotional_state: 感情状態の辞書
            dialogue_history: 対話履歴
            
        Returns:
            str: 生成された応答
        """
        # 最も強い感情を特定
        dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])[0]
        
        # 非認知能力の評価
        noncog_scores = self.evaluate_noncognitive_skills(dialogue_history)
        
        # 応答の選択
        if dominant_emotion in self.dialogue_templates:
            responses = self.dialogue_templates[dominant_emotion]
            
            # コンテキストに基づいて最適な応答を選択
            if noncog_scores['perseverance'] < 0.5 and 'frustration' in emotional_state:
                response = "一歩一歩着実に進んでいきましょう。小さな進歩も大切な成長です。"
            elif noncog_scores['curiosity'] > 0.7:
                response = "その好奇心は素晴らしいですね。もっと深く探求してみましょう。"
            else:
                response = random.choice(responses)
        else:
            response = "一緒に頑張っていきましょう！"
            
        return response
    
    def analyze_dialogue_style(self, text: str) -> Dict[str, float]:
        """
        対話スタイルを分析
        
        Args:
            text: 分析するテキスト
            
        Returns:
            Dict[str, float]: 対話スタイルの特徴量
        """
        # 文の長さや複雑さの分析
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # 感情分析
        emotions = self.emotion_classifier(text)[0]
        emotion_scores = {item['label']: item['score'] for item in emotions}
        
        return {
            'text_length': len(text),
            'avg_word_length': avg_word_length,
            'emotional_content': emotion_scores,
            'formality_score': self._calculate_formality(text)
        }
    
    def _calculate_formality(self, text: str) -> float:
        """
        テキストの形式度を計算（簡易版）
        
        Args:
            text: 分析するテキスト
            
        Returns:
            float: 形式度スコア（0-1）
        """
        formal_indicators = ['です', 'ます', 'でしょう', 'ございます']
        informal_indicators = ['だよ', 'だね', 'じゃん', 'だろ']
        
        formal_count = sum(text.count(ind) for ind in formal_indicators)
        informal_count = sum(text.count(ind) for ind in informal_indicators)
        
        total = formal_count + informal_count
        if total == 0:
            return 0.5  # デフォルト値
            
        return formal_count / total
