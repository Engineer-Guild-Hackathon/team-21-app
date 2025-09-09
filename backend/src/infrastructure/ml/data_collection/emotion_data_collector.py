import json
import os
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ...domain.models.emotion import EmotionRecord
from ...domain.models.behavior import BehaviorRecord
from ...infrastructure.database import get_db

class EmotionDataCollector:
    def __init__(self, data_dir: str = "data/collected"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # データ収集の設定
        self.collection_config = {
            "emotions": ["joy", "frustration", "concentration", "neutral"],
            "contexts": ["learning", "quest", "chat"],
            "min_samples_per_emotion": 100
        }
        
    def collect_from_database(self, db: Session) -> Dict[str, Any]:
        """データベースから感情データを収集"""
        # 感情レコードの取得
        emotion_records = db.query(EmotionRecord).all()
        
        # 行動レコードの取得（感情との関連付け用）
        behavior_records = db.query(BehaviorRecord).all()
        
        # データの整形
        collected_data = {
            "emotions": [],
            "behaviors": [],
            "correlations": []
        }
        
        for record in emotion_records:
            emotion_data = {
                "id": record.id,
                "user_id": record.user_id,
                "emotion_type": record.emotion_type,
                "intensity": record.intensity,
                "context": record.context,
                "timestamp": record.created_at.isoformat(),
                "text": record.text if hasattr(record, "text") else None
            }
            collected_data["emotions"].append(emotion_data)
        
        for record in behavior_records:
            behavior_data = {
                "id": record.id,
                "user_id": record.user_id,
                "action_type": record.action_type,
                "context": record.context,
                "timestamp": record.created_at.isoformat(),
                "success_rate": record.success_rate if hasattr(record, "success_rate") else None
            }
            collected_data["behaviors"].append(behavior_data)
        
        # 感情と行動の相関関係の分析
        df_emotions = pd.DataFrame(collected_data["emotions"])
        df_behaviors = pd.DataFrame(collected_data["behaviors"])
        
        if not df_emotions.empty and not df_behaviors.empty:
            # タイムスタンプでの結合
            df_emotions["timestamp"] = pd.to_datetime(df_emotions["timestamp"])
            df_behaviors["timestamp"] = pd.to_datetime(df_behaviors["timestamp"])
            
            # 30分以内の感情と行動を関連付け
            for _, emotion in df_emotions.iterrows():
                related_behaviors = df_behaviors[
                    (df_behaviors["user_id"] == emotion["user_id"]) &
                    (abs(df_behaviors["timestamp"] - emotion["timestamp"]) <= pd.Timedelta(minutes=30))
                ]
                
                for _, behavior in related_behaviors.iterrows():
                    correlation = {
                        "emotion_id": emotion["id"],
                        "behavior_id": behavior["id"],
                        "time_diff": (behavior["timestamp"] - emotion["timestamp"]).total_seconds()
                    }
                    collected_data["correlations"].append(correlation)
        
        return collected_data
    
    def save_collected_data(self, data: Dict[str, Any]):
        """収集したデータを保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSONファイルとして保存
        json_path = os.path.join(self.data_dir, f"emotion_data_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # CSVファイルとしても保存（分析用）
        emotions_df = pd.DataFrame(data["emotions"])
        behaviors_df = pd.DataFrame(data["behaviors"])
        correlations_df = pd.DataFrame(data["correlations"])
        
        emotions_df.to_csv(os.path.join(self.data_dir, f"emotions_{timestamp}.csv"), index=False)
        behaviors_df.to_csv(os.path.join(self.data_dir, f"behaviors_{timestamp}.csv"), index=False)
        correlations_df.to_csv(os.path.join(self.data_dir, f"correlations_{timestamp}.csv"), index=False)
    
    def analyze_data_distribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """データの分布を分析"""
        df_emotions = pd.DataFrame(data["emotions"])
        
        analysis = {
            "emotion_counts": df_emotions["emotion_type"].value_counts().to_dict(),
            "context_counts": df_emotions["context"].value_counts().to_dict(),
            "average_intensity": df_emotions["intensity"].mean(),
            "intensity_std": df_emotions["intensity"].std(),
            "total_samples": len(df_emotions),
            "unique_users": df_emotions["user_id"].nunique()
        }
        
        # 感情ごとの強度分布
        emotion_intensities = {}
        for emotion in self.collection_config["emotions"]:
            emotion_data = df_emotions[df_emotions["emotion_type"] == emotion]
            if not emotion_data.empty:
                emotion_intensities[emotion] = {
                    "mean": emotion_data["intensity"].mean(),
                    "std": emotion_data["intensity"].std(),
                    "min": emotion_data["intensity"].min(),
                    "max": emotion_data["intensity"].max()
                }
        
        analysis["emotion_intensities"] = emotion_intensities
        
        return analysis
    
    def check_data_quality(self, data: Dict[str, Any]) -> List[str]:
        """データ品質のチェック"""
        issues = []
        df_emotions = pd.DataFrame(data["emotions"])
        
        # サンプル数のチェック
        for emotion in self.collection_config["emotions"]:
            count = len(df_emotions[df_emotions["emotion_type"] == emotion])
            if count < self.collection_config["min_samples_per_emotion"]:
                issues.append(f"感情 '{emotion}' のサンプル数が不足しています（{count}/{self.collection_config['min_samples_per_emotion']}）")
        
        # 強度値の範囲チェック
        invalid_intensities = df_emotions[
            (df_emotions["intensity"] < 0) | (df_emotions["intensity"] > 1)
        ]
        if not invalid_intensities.empty:
            issues.append(f"無効な強度値が {len(invalid_intensities)} 件あります")
        
        # コンテキストの有効性チェック
        invalid_contexts = df_emotions[
            ~df_emotions["context"].isin(self.collection_config["contexts"])
        ]
        if not invalid_contexts.empty:
            issues.append(f"無効なコンテキストが {len(invalid_contexts)} 件あります")
        
        # タイムスタンプの妥当性チェック
        df_emotions["timestamp"] = pd.to_datetime(df_emotions["timestamp"])
        future_records = df_emotions[
            df_emotions["timestamp"] > datetime.now()
        ]
        if not future_records.empty:
            issues.append(f"未来の日時を持つレコードが {len(future_records)} 件あります")
        
        return issues

def collect_and_analyze_data():
    """データ収集と分析を実行"""
    collector = EmotionDataCollector()
    db = next(get_db())
    
    try:
        # データ収集
        print("データ収集を開始...")
        collected_data = collector.collect_from_database(db)
        
        # データ品質チェック
        print("\nデータ品質をチェック中...")
        quality_issues = collector.check_data_quality(collected_data)
        if quality_issues:
            print("\n検出された問題:")
            for issue in quality_issues:
                print(f"- {issue}")
        else:
            print("データ品質は良好です")
        
        # データ分布の分析
        print("\nデータ分布を分析中...")
        distribution = collector.analyze_data_distribution(collected_data)
        print("\n分析結果:")
        print(json.dumps(distribution, indent=2, ensure_ascii=False))
        
        # データの保存
        print("\nデータを保存中...")
        collector.save_collected_data(collected_data)
        print("データ収集が完了しました")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    collect_and_analyze_data()
