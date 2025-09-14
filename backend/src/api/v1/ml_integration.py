"""
MLサービスとの統合API
"""

import os
from typing import Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.security import get_current_active_user
from ...domain.models.avatar import UserStats
from ...domain.models.user import User
from ...infrastructure.database import get_db

router = APIRouter()

# MLサービスのベースURL
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml_service:8001")


async def get_ml_client():
    """MLサービス用のHTTPクライアント"""
    async with httpx.AsyncClient() as client:
        yield client


@router.post("/analyze-conversation")
async def analyze_conversation_with_ml(
    messages: List[Dict],
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    ml_client: httpx.AsyncClient = Depends(get_ml_client),
):
    """会話履歴をMLサービスで分析"""

    try:
        # MLサービスに会話分析を依頼
        analysis_request = {
            "user_id": current_user.id,
            "messages": messages,
            "current_skills": await get_current_user_skills(current_user.id, db),
        }

        response = await ml_client.post(
            f"{ML_SERVICE_URL}/analyze-conversation", json=analysis_request
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500, detail=f"ML分析エラー: {response.text}"
            )

        analysis_result = response.json()

        # ユーザー統計を更新
        await update_user_stats_from_analysis(
            current_user.id, analysis_result["skills"], db
        )

        return {
            "user_id": current_user.id,
            "skills": analysis_result["skills"],
            "feedback": analysis_result["feedback"],
            "analysis_timestamp": analysis_result["analysis_timestamp"],
        }

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"MLサービス接続エラー: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"会話分析エラー: {str(e)}")


@router.post("/update-progress")
async def update_progress_with_ml(
    activities: List[Dict],
    time_horizon_days: int = 7,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    ml_client: httpx.AsyncClient = Depends(get_ml_client),
):
    """学習活動から進捗をMLで予測・更新"""

    try:
        # 現在のスキルを取得
        current_skills = await get_current_user_skills(current_user.id, db)

        # MLサービスに進捗予測を依頼
        progress_request = {
            "user_id": current_user.id,
            "activities": activities,
            "current_skills": current_skills,
            "time_horizon_days": time_horizon_days,
        }

        response = await ml_client.post(
            f"{ML_SERVICE_URL}/predict-progress", json=progress_request
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500, detail=f"ML進捗予測エラー: {response.text}"
            )

        progress_result = response.json()

        # ユーザー統計を更新
        await update_user_stats_from_progress(
            current_user.id, progress_result["updated_skills"], db
        )

        return {
            "user_id": current_user.id,
            "prediction": progress_result["prediction"],
            "updated_skills": progress_result["updated_skills"],
            "update_timestamp": progress_result["update_timestamp"],
        }

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"MLサービス接続エラー: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"進捗更新エラー: {str(e)}")


@router.post("/generate-feedback")
async def generate_feedback_with_ml(
    message: str,
    context: Optional[Dict] = None,
    current_user: User = Depends(get_current_active_user),
    ml_client: httpx.AsyncClient = Depends(get_ml_client),
):
    """MLサービスでリアルタイムフィードバック生成"""

    try:
        feedback_request = {
            "user_id": current_user.id,
            "message": message,
            "context": context,
        }

        response = await ml_client.post(
            f"{ML_SERVICE_URL}/generate-feedback", json=feedback_request
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500, detail=f"MLフィードバック生成エラー: {response.text}"
            )

        return response.json()

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"MLサービス接続エラー: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"フィードバック生成エラー: {str(e)}"
        )


async def get_current_user_skills(user_id: int, db: AsyncSession) -> Dict[str, float]:
    """ユーザーの現在のスキルを取得"""

    stmt = select(UserStats).where(UserStats.user_id == user_id)
    result = await db.execute(stmt)
    stats = result.scalar_one_or_none()

    if not stats:
        # デフォルトスキル値
        return {
            "grit": 2.0,
            "collaboration": 2.0,
            "self_regulation": 2.0,
            "emotional_intelligence": 2.0,
            "confidence": 2.0,
        }

    return {
        "grit": stats.grit_level,
        "collaboration": stats.collaboration_level,
        "self_regulation": stats.self_regulation_level,
        "emotional_intelligence": stats.emotional_intelligence_level,
        "confidence": 2.0,  # デフォルト値（まだ実装されていない）
    }


async def update_user_stats_from_analysis(user_id: int, skills: Dict, db: AsyncSession):
    """ML分析結果からユーザー統計を更新"""

    stmt = select(UserStats).where(UserStats.user_id == user_id)
    result = await db.execute(stmt)
    stats = result.scalar_one_or_none()

    if not stats:
        stats = UserStats(user_id=user_id)
        db.add(stats)

    # スキル値を更新
    stats.grit_level = skills.get("grit", stats.grit_level)
    stats.collaboration_level = skills.get("collaboration", stats.collaboration_level)
    stats.self_regulation_level = skills.get(
        "self_regulation", stats.self_regulation_level
    )
    stats.emotional_intelligence_level = skills.get(
        "emotional_intelligence", stats.emotional_intelligence_level
    )

    await db.commit()
    await db.refresh(stats)


async def update_user_stats_from_progress(
    user_id: int, updated_skills: Dict, db: AsyncSession
):
    """ML進捗予測結果からユーザー統計を更新"""

    stmt = select(UserStats).where(UserStats.user_id == user_id)
    result = await db.execute(stmt)
    stats = result.scalar_one_or_none()

    if not stats:
        stats = UserStats(user_id=user_id)
        db.add(stats)

    # スキル値を更新
    stats.grit_level = updated_skills.get("grit", stats.grit_level)
    stats.collaboration_level = updated_skills.get(
        "collaboration", stats.collaboration_level
    )
    stats.self_regulation_level = updated_skills.get(
        "self_regulation", stats.self_regulation_level
    )
    stats.emotional_intelligence_level = updated_skills.get(
        "emotional_intelligence", stats.emotional_intelligence_level
    )

    await db.commit()
    await db.refresh(stats)


@router.get("/health")
async def ml_service_health_check(
    ml_client: httpx.AsyncClient = Depends(get_ml_client),
):
    """MLサービスのヘルスチェック"""

    try:
        response = await ml_client.get(f"{ML_SERVICE_URL}/health")

        if response.status_code == 200:
            return {"status": "healthy", "ml_service": "connected"}
        else:
            return {"status": "unhealthy", "ml_service": "error"}

    except httpx.RequestError:
        return {"status": "unhealthy", "ml_service": "disconnected"}
