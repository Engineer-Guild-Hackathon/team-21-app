import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import Dict, List, Tuple
import json
import logging

class EmotionAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.bert = BertModel.from_pretrained("cl-tohoku/bert-base-japanese").to(self.device)
        self.emotion_classifier = EmotionClassifier(self.bert.config.hidden_size).to(self.device)
        self.load_models()

    def load_models(self):
        try:
            # 感情分類モデルの読み込み
            self.emotion_classifier.load_state_dict(
                torch.load("models/emotion_classifier.pth", map_location=self.device)
            )
            self.emotion_classifier.eval()
        except FileNotFoundError:
            logging.warning("事前学習済みモデルが見つかりません。デフォルトのモデルを使用します。")

    def analyze_text(self, text: str) -> Dict[str, float]:
        # テキストのトークン化
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # BERTによる特徴抽出
        with torch.no_grad():
            outputs = self.bert(**inputs)
            pooled_output = outputs.pooler_output

        # 感情スコアの計算
        emotion_scores = self.emotion_classifier(pooled_output)
        emotion_scores = F.softmax(emotion_scores, dim=1)

        # 結果の整形
        emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
        return {
            emotion: score.item()
            for emotion, score in zip(emotions, emotion_scores[0])
        }

    def analyze_emotion_log(self, emotion_log: dict) -> dict:
        # テキスト分析
        text_analysis = self.analyze_text(emotion_log["trigger"])
        
        # 報告された感情の強度を考慮
        reported_intensity = emotion_log["intensity"] / 10.0  # 0-1のスケールに正規化
        
        # ストレスレベルの計算
        stress_level = self.calculate_stress_level(
            text_analysis,
            reported_intensity,
            emotion_log["emotion"]
        )
        
        # 最も強い感情の特定
        dominant_emotion = max(text_analysis.items(), key=lambda x: x[1])[0]
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores": text_analysis,
            "stress_level": stress_level
        }

    def calculate_stress_level(
        self,
        emotion_scores: Dict[str, float],
        reported_intensity: float,
        reported_emotion: str
    ) -> float:
        # ストレス関連の感情のウェイト
        stress_weights = {
            "anger": 0.8,
            "fear": 0.7,
            "sadness": 0.6,
            "disgust": 0.5,
            "surprise": 0.3,
            "anticipation": 0.2,
            "joy": -0.3,
            "trust": -0.2
        }
        
        # テキストベースのストレススコア
        text_stress = sum(
            score * stress_weights[emotion]
            for emotion, score in emotion_scores.items()
        )
        
        # 報告された感情のストレス影響
        reported_stress = stress_weights.get(reported_emotion, 0) * reported_intensity
        
        # 総合的なストレスレベル（0-1の範囲に正規化）
        stress_level = (text_stress + reported_stress) / 2
        return max(0, min(1, (stress_level + 1) / 2))

class EmotionClassifier(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 8)  # 8感情分類

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.classifier(x)

class EmotionPatternAnalyzer:
    def __init__(self):
        self.pattern_window = 10  # パターン検出の時間窓
        self.emotion_sequences = []

    def add_emotion_record(self, emotion: str, intensity: float, timestamp: float):
        self.emotion_sequences.append({
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": timestamp
        })
        
        # 時間窓を超えた古いレコードを削除
        current_time = timestamp
        self.emotion_sequences = [
            record for record in self.emotion_sequences
            if current_time - record["timestamp"] <= self.pattern_window
        ]

    def analyze_patterns(self) -> Dict[str, List[Dict]]:
        if len(self.emotion_sequences) < 3:
            return {"patterns": [], "triggers": []}

        # 感情の遷移パターンを検出
        transitions = []
        for i in range(len(self.emotion_sequences) - 1):
            current = self.emotion_sequences[i]
            next_emotion = self.emotion_sequences[i + 1]
            transitions.append({
                "from": current["emotion"],
                "to": next_emotion["emotion"],
                "time_diff": next_emotion["timestamp"] - current["timestamp"]
            })

        # 頻出パターンの特定
        pattern_counts = {}
        for i in range(len(transitions) - 1):
            pattern = f"{transitions[i]['from']}->{transitions[i]['to']}"
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # 上位のパターンを抽出
        significant_patterns = sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        return {
            "patterns": [
                {
                    "sequence": pattern,
                    "frequency": count
                }
                for pattern, count in significant_patterns
            ],
            "triggers": self.identify_triggers()
        }

    def identify_triggers(self) -> List[Dict]:
        if len(self.emotion_sequences) < 2:
            return []

        # 感情の急激な変化を検出
        triggers = []
        for i in range(len(self.emotion_sequences) - 1):
            current = self.emotion_sequences[i]
            next_emotion = self.emotion_sequences[i + 1]
            
            # 強度の変化が大きい場合
            if abs(next_emotion["intensity"] - current["intensity"]) > 0.5:
                triggers.append({
                    "from_emotion": current["emotion"],
                    "to_emotion": next_emotion["emotion"],
                    "intensity_change": next_emotion["intensity"] - current["intensity"],
                    "timestamp": next_emotion["timestamp"]
                })

        return sorted(triggers, key=lambda x: abs(x["intensity_change"]), reverse=True)[:3]
