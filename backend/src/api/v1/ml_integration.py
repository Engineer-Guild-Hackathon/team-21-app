"""
MLサービスとの統合API
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.security import get_current_active_user
from ...domain.models.avatar import UserStats
from ...domain.models.chat import ChatMessage, ChatSession
from ...domain.models.quest import QuestProgress, QuestStatus
from ...domain.models.user import User
from ...infrastructure.database import get_db

router = APIRouter()

# MLサービスのベースURL
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml_service:8001")


async def get_ml_client():
    """MLサービス用のHTTPクライアント"""
    async with httpx.AsyncClient() as client:
        yield client


async def get_user_conversation_history(
    user_id: int, db: AsyncSession, limit: int = 50
) -> List[Dict]:
    """ユーザーの会話履歴をデータベースから取得"""

    # 最新のチャットセッションから会話を取得
    stmt = (
        select(ChatMessage)
        .join(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )

    result = await db.execute(stmt)
    messages = result.scalars().all()

    # 時系列順に並び替え（古い順）
    messages.reverse()

    # ML分析用の形式に変換
    conversation_history = []
    for message in messages:
        conversation_history.append(
            {
                "role": message.role,
                "content": message.content,
                "timestamp": message.created_at.isoformat(),
                "session_id": message.session_id,
            }
        )

    return conversation_history


async def get_user_quest_data(user_id: int, db: AsyncSession) -> Dict:
    """ユーザーのクエストデータを取得"""

    # 完了したクエスト数
    completed_count = (
        await db.scalar(
            select(func.count(QuestProgress.id)).where(
                QuestProgress.user_id == user_id,
                QuestProgress.status == QuestStatus.COMPLETED,
            )
        )
        or 0
    )

    # 進行中のクエスト数
    in_progress_count = (
        await db.scalar(
            select(func.count(QuestProgress.id)).where(
                QuestProgress.user_id == user_id,
                QuestProgress.status == QuestStatus.IN_PROGRESS,
            )
        )
        or 0
    )

    # 連続達成日数の最大値
    max_streak = (
        await db.scalar(
            select(func.max(QuestProgress.streak_count)).where(
                QuestProgress.user_id == user_id
            )
        )
        or 0
    )

    # 最近のクエスト活動（過去30日）
    recent_quests = await db.execute(
        select(QuestProgress)
        .where(
            QuestProgress.user_id == user_id,
            QuestProgress.completed_date >= func.now() - func.interval("30 days"),
        )
        .order_by(QuestProgress.completed_date.desc())
        .limit(10)
    )
    recent_quests_data = recent_quests.scalars().all()

    # クエストタイプ別の完了数
    quest_types = {}
    for quest_progress in recent_quests_data:
        if quest_progress.quest:
            quest_type = quest_progress.quest.quest_type
            quest_types[quest_type] = quest_types.get(quest_type, 0) + 1

    return {
        "total_completed": completed_count,
        "in_progress": in_progress_count,
        "max_streak_days": max_streak,
        "recent_quest_types": quest_types,
        "recent_activity_count": len(recent_quests_data),
    }


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


@router.post("/analyze-from-database")
async def analyze_conversation_from_database(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    ml_client: httpx.AsyncClient = Depends(get_ml_client),
):
    """データベースから会話履歴を取得してML分析を実行"""
    try:
        # データベースから会話履歴を取得
        conversation_history = await get_user_conversation_history(
            current_user.id, db, limit=50
        )

        # クエストデータを取得
        quest_data = await get_user_quest_data(current_user.id, db)

        if not conversation_history and quest_data["total_completed"] == 0:
            return {"message": "分析対象のデータがありません"}

        print(f"取得した会話履歴数: {len(conversation_history)}")
        print(f"完了したクエスト数: {quest_data['total_completed']}")

        # ML分析リクエストの準備
        analysis_request = {
            "user_id": current_user.id,
            "messages": conversation_history,
            "quest_data": quest_data,  # クエストデータを追加
            "current_skills": await get_current_user_skills(current_user.id, db),
            "analysis_type": "comprehensive_analysis",  # 包括的分析
            "include_context": True,  # 文脈を含めた分析
        }

        # MLサービスにリクエスト送信
        response = await ml_client.post(
            f"{ML_SERVICE_URL}/analyze-conversation",
            json=analysis_request,
            timeout=60.0,  # より長いタイムアウト
        )

        if response.status_code == 200:
            analysis_result = response.json()
            print(f"データベースベースML分析結果: {analysis_result}")

            # ユーザー統計を更新
            await update_user_stats_from_analysis(
                current_user.id, analysis_result["skills"], db
            )

            return {
                "user_id": current_user.id,
                "skills": analysis_result["skills"],
                "feedback": analysis_result["feedback"],
                "analysis_timestamp": analysis_result["analysis_timestamp"],
                "conversation_count": len(conversation_history),
                "quest_data": quest_data,  # クエストデータも含める
                "message": "包括的ML分析が完了しました",
            }
        else:
            print(
                f"データベースベースML分析エラー: {response.status_code} - {response.text}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"データベースベースML分析エラー: {response.text}",
            )

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"MLサービス接続エラー: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"データベースベースML分析エラー: {str(e)}"
        )


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
        # 新しいユーザーには初期値（1.0）を返す
        return {
            "grit": 1.0,
            "collaboration": 1.0,
            "self_regulation": 1.0,
            "emotional_intelligence": 1.0,
            "confidence": 1.0,
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


@router.get("/latest-analysis")
async def get_latest_ml_analysis(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """最新のML分析結果を取得"""

    try:
        # 現在のユーザースキルを取得（これが最新のML分析結果）
        current_skills = await get_current_user_skills(current_user.id, db)

        # 実際のML分析結果として、現在のスキル値を使用
        latest_analysis = {
            "user_id": current_user.id,
            "skills": current_skills,
            "feedback": generate_feedback_from_skills(current_skills),
            "analysis_timestamp": datetime.now().isoformat(),
        }

        print(
            f"🔍 最新ML分析結果取得: ユーザー{current_user.id}, スキル: {current_skills}"
        )

        return latest_analysis

    except Exception as e:
        print(f"❌ ML分析結果取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析結果取得エラー: {str(e)}")


def generate_feedback_from_skills(skills: Dict[str, float]) -> str:
    """スキルデータからフィードバックを生成"""

    feedbacks = []

    # 全てのスキルが初期値（1.0）の場合、新規ユーザー向けメッセージ
    if all(value == 1.0 for key, value in skills.items() if key != "confidence"):
        return "🎯 学習を始めましょう！AIチャットで質問したり、クエストに挑戦したりして、スキルを向上させていきましょう。"

    if skills.get("grit", 0) >= 4.0:
        feedbacks.append(
            "🌟 素晴らしいやり抜く力を持っています！困難な課題にも諦めずに取り組む姿勢が見られます。"
        )
    elif skills.get("grit", 0) >= 3.0:
        feedbacks.append(
            "👍 やり抜く力が向上しています。目標を設定して継続的に取り組んでみましょう。"
        )
    else:
        feedbacks.append(
            "💪 やり抜く力を鍛えるために、小さな目標から始めて達成感を積み重ねていきましょう。"
        )

    if skills.get("collaboration", 0) >= 4.0:
        feedbacks.append(
            "🤝 協調性がとても高いです！他者との協力を大切にしていますね。"
        )
    elif skills.get("collaboration", 0) >= 3.0:
        feedbacks.append(
            "👥 協調性が育っています。グループ学習やペア学習を活用してみましょう。"
        )
    else:
        feedbacks.append(
            "🤝 協調性を高めるために、友達と一緒に勉強したり、質問を積極的にしてみましょう。"
        )

    if skills.get("self_regulation", 0) >= 4.0:
        feedbacks.append(
            "🎯 自己制御力が優れています！計画的に学習を進められています。"
        )
    elif skills.get("self_regulation", 0) >= 3.0:
        feedbacks.append(
            "📝 自己制御力が向上しています。学習計画を立てて実行してみましょう。"
        )
    else:
        feedbacks.append(
            "⏰ 自己制御力を高めるために、学習時間を決めて集中して取り組んでみましょう。"
        )

    if skills.get("emotional_intelligence", 0) >= 4.0:
        feedbacks.append(
            "💝 感情知能が高いです！自分の感情を理解し、適切に表現できています。"
        )
    elif skills.get("emotional_intelligence", 0) >= 3.0:
        feedbacks.append(
            "😊 感情知能が育っています。感情を言葉で表現する練習をしてみましょう。"
        )
    else:
        feedbacks.append(
            "💭 感情知能を高めるために、自分の気持ちを振り返る時間を作ってみましょう。"
        )

    return "\n\n".join(feedbacks)


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
