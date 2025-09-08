from datetime import datetime
from typing import List

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.domain.models import EmotionRecord
from src.infrastructure.database import get_db

router = APIRouter()


class EmotionLog(BaseModel):
    emotion: str
    intensity: float
    trigger: str
    timestamp: datetime


class EmotionAnalysis(BaseModel):
    dominant_emotion: str
    emotion_scores: dict
    stress_level: float
    recommendations: List[str]


@router.post("/analyze", response_model=EmotionAnalysis)
async def analyze_emotions(emotion_log: EmotionLog, db: Session = Depends(get_db)):
    # 感情分析の実行
    analysis_result = await analyze_emotion_data(emotion_log)

    # 分析結果の保存
    db_record = EmotionRecord(
        emotion_type=analysis_result.dominant_emotion,
        confidence=max(analysis_result.emotion_scores.values()),
        timestamp=emotion_log.timestamp,
    )
    db.add(db_record)
    db.commit()

    return analysis_result


async def analyze_emotion_data(emotion_log: EmotionLog) -> EmotionAnalysis:
    # MLサービスへのリクエスト
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://ml_service:8001/emotions/analyze", json=emotion_log.dict()
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="感情分析に失敗しました")

        analysis = response.json()

    # 推奨アクションの生成
    recommendations = generate_recommendations(
        analysis["dominant_emotion"], analysis["stress_level"]
    )

    return EmotionAnalysis(
        dominant_emotion=analysis["dominant_emotion"],
        emotion_scores=analysis["emotion_scores"],
        stress_level=analysis["stress_level"],
        recommendations=recommendations,
    )


def generate_recommendations(emotion: str, stress_level: float) -> List[str]:
    recommendations = []

    if stress_level > 0.7:
        recommendations.extend(
            [
                "深呼吸を行い、リラックスしましょう",
                "短い休憩を取ることをお勧めします",
                "軽い運動で気分転換を図りましょう",
            ]
        )

    if emotion == "anger":
        recommendations.extend(
            [
                "その場を離れて冷静になる時間を取りましょう",
                "感情を言葉で表現してみましょう",
                "相手の立場に立って考えてみましょう",
            ]
        )
    elif emotion == "sadness":
        recommendations.extend(
            [
                "信頼できる人に気持ちを話してみましょう",
                "好きな活動で気分を上げてみましょう",
                "小さな目標を立てて達成感を得ましょう",
            ]
        )
    elif emotion == "fear":
        recommendations.extend(
            [
                "不安の原因を具体的に書き出してみましょう",
                "対処可能な小さな課題に分解してみましょう",
                "過去の成功体験を思い出してみましょう",
            ]
        )

    return recommendations[:3]  # 最大3つの推奨を返す


@router.get("/history/{user_id}", response_model=List[EmotionLog])
async def get_emotion_history(user_id: str, db: Session = Depends(get_db)):
    records = (
        db.query(EmotionRecord)
        .filter(EmotionRecord.user_id == user_id)
        .order_by(EmotionRecord.timestamp.desc())
        .limit(10)
        .all()
    )

    return records


@router.get("/trends/{user_id}")
async def get_emotion_trends(user_id: str, db: Session = Depends(get_db)):
    # 感情の傾向分析
    records = (
        db.query(EmotionRecord)
        .filter(EmotionRecord.user_id == user_id)
        .order_by(EmotionRecord.timestamp.desc())
        .limit(100)
        .all()
    )

    # トレンド分析の実行
    trends = analyze_trends(records)

    return {
        "emotion_distribution": trends["distribution"],
        "stress_trend": trends["stress_trend"],
        "improvement_areas": trends["improvement_areas"],
    }


def analyze_trends(records: List[EmotionRecord]) -> dict:
    # 感情の分布を計算
    emotion_counts = {}
    for record in records:
        emotion_counts[record.emotion_type] = (
            emotion_counts.get(record.emotion_type, 0) + 1
        )

    # ストレスレベルの傾向
    stress_trend = []
    for record in records:
        if record.emotion_type in ["stress", "anxiety", "anger"]:
            stress_trend.append(
                {"timestamp": record.timestamp, "level": record.confidence}
            )

    # 改善が必要な領域を特定
    improvement_areas = []
    if len(records) > 0:
        negative_emotions = ["stress", "anxiety", "anger", "sadness"]
        for emotion in negative_emotions:
            count = emotion_counts.get(emotion, 0)
            if count / len(records) > 0.3:  # 30%以上の頻度で出現
                improvement_areas.append(
                    {"emotion": emotion, "frequency": count / len(records)}
                )

    return {
        "distribution": emotion_counts,
        "stress_trend": stress_trend,
        "improvement_areas": improvement_areas,
    }
