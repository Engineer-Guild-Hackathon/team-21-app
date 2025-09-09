from typing import Dict, Any
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from PIL import Image
import cv2

class EmotionAnalyzer:
    def __init__(self):
        # テキスト感情分析のモデル
        self.text_tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.text_model = BertForSequenceClassification.from_pretrained(
            'cl-tohoku/bert-base-japanese-whole-word-masking',
            num_labels=7  # 感情カテゴリの数
        )
        
        # デバイスの設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_model.to(self.device)
        
        # 感情カテゴリ
        self.emotions = [
            "joy", "sadness", "anger", "fear",
            "surprise", "frustration", "concentration"
        ]
        
        # 顔検出のための設定
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """テキストから感情を分析"""
        # テキストの前処理
        inputs = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 感情分析の実行
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # 結果の整形
        emotion_scores = probabilities[0].cpu().numpy()
        emotions_dict = dict(zip(self.emotions, emotion_scores))
        
        # 最も強い感情を特定
        dominant_emotion = max(emotions_dict.items(), key=lambda x: x[1])
        
        return {
            "emotion": dominant_emotion[0],
            "intensity": float(dominant_emotion[1]),
            "emotions": emotions_dict
        }
    
    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """画像から感情を分析"""
        # 画像をOpenCV形式に変換
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 顔検出
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return {
                "emotion": "neutral",
                "intensity": 0.0,
                "emotions": {emotion: 0.0 for emotion in self.emotions}
            }
        
        # 最も大きい顔を分析
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        face_img = gray[y:y+h, x:x+w]
        
        # ここで実際の感情分析モデルを使用する
        # 現在はダミーの実装
        emotions_dict = {
            "joy": 0.6,
            "sadness": 0.1,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.2,
            "frustration": 0.0,
            "concentration": 0.1
        }
        
        dominant_emotion = max(emotions_dict.items(), key=lambda x: x[1])
        
        return {
            "emotion": dominant_emotion[0],
            "intensity": float(dominant_emotion[1]),
            "emotions": emotions_dict,
            "face_location": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        }
    
    def combine_analysis(
        self,
        text_result: Dict[str, Any],
        image_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """テキストと画像の分析結果を組み合わせる"""
        if not image_result:
            return text_result
        
        # 感情スコアの重み付け平均を計算
        combined_emotions = {}
        for emotion in self.emotions:
            text_score = text_result["emotions"].get(emotion, 0.0)
            image_score = image_result["emotions"].get(emotion, 0.0)
            # テキストと画像の重みを7:3に設定
            combined_emotions[emotion] = 0.7 * text_score + 0.3 * image_score
        
        dominant_emotion = max(combined_emotions.items(), key=lambda x: x[1])
        
        return {
            "emotion": dominant_emotion[0],
            "intensity": float(dominant_emotion[1]),
            "emotions": combined_emotions,
            "text_analysis": text_result,
            "image_analysis": image_result
        }
