import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass
from enum import Enum

class EmotionCategory(Enum):
    """感情カテゴリの定義"""
    MOTIVATED = "motivated"       # やる気がある
    FRUSTRATED = "frustrated"     # 挫折している
    CONFUSED = "confused"        # 混乱している
    SATISFIED = "satisfied"      # 満足している
    ANXIOUS = "anxious"         # 不安である
    CONFIDENT = "confident"     # 自信がある
    BORED = "bored"            # 退屈している

@dataclass
class EmotionAnalysisResult:
    """感情分析の結果を格納するクラス"""
    primary_emotion: EmotionCategory
    emotion_scores: Dict[EmotionCategory, float]
    confidence: float
    text_features: Optional[torch.Tensor] = None
    behavioral_features: Optional[torch.Tensor] = None

class EmotionAnalyzer:
    """感情分析モジュール"""
    
    def __init__(self, model_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # BERTモデルとトークナイザーの初期化
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # 感情分類層の初期化
        self.emotion_classifier = EmotionClassifier(
            bert_hidden_size=768,
            behavioral_feature_size=10,
            num_emotions=len(EmotionCategory)
        ).to(self.device)
        
        # ロガーの設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_emotion(self,
                       text: Optional[str] = None,
                       behavioral_data: Optional[Dict[str, float]] = None) -> EmotionAnalysisResult:
        """テキストと行動データから感情を分析"""
        with torch.no_grad():
            # テキストの特徴抽出
            text_features = self._extract_text_features(text) if text else None
            
            # 行動データの特徴抽出
            behavioral_features = self._extract_behavioral_features(behavioral_data) if behavioral_data else None
            
            # 感情の分類
            emotion_scores = self._classify_emotion(text_features, behavioral_features)
            
            # 最も確信度の高い感情の特定
            primary_emotion_idx = torch.argmax(emotion_scores).item()
            primary_emotion = list(EmotionCategory)[primary_emotion_idx]
            
            # 確信度の計算
            confidence = torch.softmax(emotion_scores, dim=0)[primary_emotion_idx].item()
            
            # 感情スコアの辞書を作成
            emotion_dict = {
                emotion: score.item()
                for emotion, score in zip(EmotionCategory, emotion_scores)
            }
            
            return EmotionAnalysisResult(
                primary_emotion=primary_emotion,
                emotion_scores=emotion_dict,
                confidence=confidence,
                text_features=text_features,
                behavioral_features=behavioral_features
            )
    
    def _extract_text_features(self, text: str) -> torch.Tensor:
        """テキストから特徴を抽出"""
        # テキストのトークン化
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # BERTによる特徴抽出
        outputs = self.bert_model(**inputs)
        
        # [CLS]トークンの出力を使用
        return outputs.last_hidden_state[:, 0, :]
    
    def _extract_behavioral_features(self, behavioral_data: Dict[str, float]) -> torch.Tensor:
        """行動データから特徴を抽出"""
        # 行動特徴の正規化と変換
        features = [
            behavioral_data.get('challenge_attempts', 0) / 10.0,  # チャレンジ試行回数
            behavioral_data.get('success_rate', 0.5),            # 成功率
            behavioral_data.get('response_time', 0) / 60.0,      # 応答時間（分）
            behavioral_data.get('help_requests', 0) / 5.0,       # ヘルプ要求回数
            behavioral_data.get('task_switches', 0) / 5.0,       # タスク切り替え回数
            behavioral_data.get('focus_duration', 0) / 30.0,     # 集中持続時間（分）
            behavioral_data.get('error_rate', 0),                # エラー率
            behavioral_data.get('correction_rate', 0),           # 修正率
            behavioral_data.get('exploration_rate', 0),          # 探索率
            behavioral_data.get('completion_rate', 0)            # 完了率
        ]
        
        return torch.tensor(features, dtype=torch.float32).to(self.device)

class EmotionClassifier(nn.Module):
    """感情分類モデル"""
    
    def __init__(self,
                 bert_hidden_size: int,
                 behavioral_feature_size: int,
                 num_emotions: int,
                 hidden_size: int = 256):
        super(EmotionClassifier, self).__init__()
        
        # テキスト特徴の処理層
        self.text_processor = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 行動特徴の処理層
        self.behavioral_processor = nn.Sequential(
            nn.Linear(behavioral_feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 特徴の結合と感情分類
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_emotions)
        )
        
        # 注意機構
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
    
    def forward(self,
                text_features: Optional[torch.Tensor] = None,
                behavioral_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """順伝播"""
        batch_size = 1
        hidden_size = self.text_processor[0].out_features
        
        # 特徴ベクトルの初期化
        if text_features is None:
            text_features = torch.zeros((batch_size, hidden_size)).to(next(self.parameters()).device)
        else:
            text_features = self.text_processor(text_features)
        
        if behavioral_features is None:
            behavioral_features = torch.zeros((batch_size, hidden_size)).to(next(self.parameters()).device)
        else:
            behavioral_features = self.behavioral_processor(behavioral_features)
        
        # 注意機構による特徴の統合
        text_features = text_features.unsqueeze(0)
        behavioral_features = behavioral_features.unsqueeze(0)
        
        attended_features, _ = self.attention(
            text_features,
            behavioral_features,
            behavioral_features
        )
        
        # 特徴の結合
        combined_features = torch.cat([
            attended_features.squeeze(0),
            behavioral_features.squeeze(0)
        ], dim=-1)
        
        # 感情の分類
        return self.classifier(combined_features).squeeze(0)

class EmotionTracker:
    """感情の時系列変化を追跡するクラス"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.emotion_history: List[EmotionAnalysisResult] = []
        self.behavioral_history: List[Dict[str, float]] = []
    
    def add_emotion(self, emotion_result: EmotionAnalysisResult,
                   behavioral_data: Optional[Dict[str, float]] = None):
        """感情分析結果の追加"""
        self.emotion_history.append(emotion_result)
        if len(self.emotion_history) > self.window_size:
            self.emotion_history.pop(0)
        
        if behavioral_data:
            self.behavioral_history.append(behavioral_data)
            if len(self.behavioral_history) > self.window_size:
                self.behavioral_history.pop(0)
    
    def get_emotion_trend(self) -> Dict[str, float]:
        """感情の傾向分析"""
        if not self.emotion_history:
            return {}
        
        # 各感情カテゴリーの平均スコアを計算
        emotion_scores = {emotion: [] for emotion in EmotionCategory}
        for result in self.emotion_history:
            for emotion, score in result.emotion_scores.items():
                emotion_scores[emotion].append(score)
        
        return {
            emotion.value: np.mean(scores)
            for emotion, scores in emotion_scores.items()
        }
    
    def get_behavioral_trend(self) -> Dict[str, float]:
        """行動傾向の分析"""
        if not self.behavioral_history:
            return {}
        
        # 各行動指標の平均値を計算
        behavioral_metrics = {}
        for metric in self.behavioral_history[0].keys():
            values = [data[metric] for data in self.behavioral_history]
            behavioral_metrics[metric] = np.mean(values)
        
        return behavioral_metrics
    
    def detect_emotional_changes(self) -> List[Dict[str, any]]:
        """感情の変化点を検出"""
        if len(self.emotion_history) < 2:
            return []
        
        changes = []
        for i in range(1, len(self.emotion_history)):
            prev_emotion = self.emotion_history[i-1].primary_emotion
            curr_emotion = self.emotion_history[i].primary_emotion
            
            if prev_emotion != curr_emotion:
                changes.append({
                    'from_emotion': prev_emotion.value,
                    'to_emotion': curr_emotion.value,
                    'confidence': self.emotion_history[i].confidence,
                    'index': i
                })
        
        return changes
    
    def to_json(self) -> str:
        """感情履歴のJSON形式での出力"""
        history = []
        for i, emotion_result in enumerate(self.emotion_history):
            entry = {
                'timestamp': i,
                'primary_emotion': emotion_result.primary_emotion.value,
                'confidence': emotion_result.confidence,
                'emotion_scores': {
                    emotion.value: score
                    for emotion, score in emotion_result.emotion_scores.items()
                }
            }
            
            if i < len(self.behavioral_history):
                entry['behavioral_data'] = self.behavioral_history[i]
            
            history.append(entry)
        
        return json.dumps(history, indent=2)
