import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...infrastructure.database import get_db
from ...infrastructure.kafka_consumer import KafkaEventConsumer
from ...infrastructure.kafka_producer import KafkaEventProducer
from ...infrastructure.repositories.learning_repository import LearningEventRepository
from ...infrastructure.sse_manager import sse_manager

router = APIRouter()


@router.get("/")
async def get_learning_status() -> dict[str, str]:
    """学習状態を取得"""
    return {"status": "実装予定"}


# --- MVP: 学習行動イベント受付と簡易集計 ------------------------------


class LearnActionEvent(BaseModel):
    event_id: str = Field(..., max_length=128)
    user_id: str = Field(..., max_length=128)
    session_id: str = Field(..., max_length=128)
    action: str = Field(..., pattern="^(answer_submit|hint_request|retry|give_up)$")
    think_time_ms: int = Field(..., ge=0, le=120000)
    success: Optional[bool] = None
    difficulty: Optional[str] = Field(None, pattern="^(easy|normal|challenge)$")
    created_at: Optional[datetime] = None


class NonCogSummary(BaseModel):
    user_id: str
    retry_count: int
    avg_think_time_ms: float
    re_challenge_rate: float
    grit_score: int
    srl_score: int
    updated_at: datetime


_USER_AGGREGATES: dict[str, dict] = {}

# Kafkaコンシューマ（グローバル）
_kafka_consumer: Optional[KafkaEventConsumer] = None


def _compute_scores(
    retry_count: int, give_up_count: int, avg_think_time_ms: float
) -> tuple[int, int]:
    total_hard = max(retry_count + give_up_count, 1)
    re_challenge_rate = retry_count / total_hard
    # 簡易ヒューリスティクス: やり抜く力は再挑戦率重視、SRLは思考時間も考慮
    grit = int(max(0, min(100, 40 + re_challenge_rate * 60)))
    # 平均思考時間が長いほど自己調整が高いと仮定（上限100秒換算）
    srl = int(max(0, min(100, 30 + (avg_think_time_ms / 100000) * 70)))
    return grit, srl


def _handle_kafka_event(event_data: dict) -> None:
    """Kafkaイベント受信時のハンドラ"""
    try:
        user_id = event_data.get("user_id")
        if not user_id:
            return

        # インメモリ集計を更新
        agg = _USER_AGGREGATES.get(user_id)
        if agg:
            # 既存の集計ロジックを適用
            action = event_data.get("action", "")
            think_time_ms = event_data.get("think_time_ms", 0)

            if action == "retry":
                agg["retry_count"] += 1
            if action == "give_up":
                agg["give_up_count"] += 1

            agg["total_think_ms"] += float(think_time_ms)
            agg["event_count"] += 1
            agg["avg_think_time_ms"] = agg["total_think_ms"] / max(
                agg["event_count"], 1
            )
            agg["updated_at"] = datetime.utcnow()

            # スコアを計算
            grit, srl = _compute_scores(
                agg["retry_count"], agg["give_up_count"], agg["avg_think_time_ms"]
            )

            # SSEでクライアントに通知
            summary_data = {
                "type": "summary_update",
                "user_id": user_id,
                "retry_count": agg["retry_count"],
                "avg_think_time_ms": agg["avg_think_time_ms"],
                "re_challenge_rate": agg["retry_count"] / max(agg["event_count"], 1),
                "grit_score": grit,
                "srl_score": srl,
                "updated_at": agg["updated_at"].isoformat(),
            }

            # 非同期でSSEブロードキャスト
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                loop.create_task(sse_manager.broadcast(summary_data))
            except Exception as e:
                logging.exception("sse_broadcast_task_error: %s", e)

    except Exception as e:
        logging.exception("kafka_event_handler_error: %s", e)


def get_kafka_consumer() -> KafkaEventConsumer:
    """Kafkaコンシューマを取得（シングルトン）"""
    global _kafka_consumer
    if _kafka_consumer is None:
        _kafka_consumer = KafkaEventConsumer()
        _kafka_consumer.add_event_handler(_handle_kafka_event)
    return _kafka_consumer


@router.post("/events/learn-action", status_code=201)
async def post_learn_action(
    event: LearnActionEvent,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """学習行動イベントを受信してインメモリで集計（MVP）"""
    print(f"DEBUG: post_learn_action called with event_id={event.event_id}")
    now = datetime.utcnow()
    created = event.created_at or now

    agg = _USER_AGGREGATES.setdefault(
        event.user_id,
        {
            "retry_count": 0,
            "give_up_count": 0,
            "total_think_ms": 0.0,
            "event_count": 0,
            "avg_think_time_ms": 0.0,
            "updated_at": now,
        },
    )

    if event.action == "retry":
        agg["retry_count"] += 1
    if event.action == "give_up":
        agg["give_up_count"] += 1

    agg["total_think_ms"] += float(event.think_time_ms)
    agg["event_count"] += 1
    agg["avg_think_time_ms"] = agg["total_think_ms"] / max(agg["event_count"], 1)
    agg["updated_at"] = created

    print(f"DEBUG: Event processed event_id={event.event_id}")

    # 環境変数で永続化を段階導入（デフォルト: OFF）
    persist_flag = os.getenv("PERSIST_LEARN_EVENTS", "false").lower()
    logging.info("PERSIST_LEARN_EVENTS=%s", persist_flag)
    if persist_flag == "true":
        logging.info("persist branch entered for user_id=%s", event.user_id)
        try:
            numeric_user_id = int(event.user_id)
            repo = LearningEventRepository()
            rec = await repo.create_from_learn_action(
                db,
                user_id=numeric_user_id,
                action=event.action,
                think_time_ms=event.think_time_ms,
                success=event.success,
                created_at=created,
            )
            logging.info(
                "persist success for user_id=%s record_id=%s",
                numeric_user_id,
                getattr(rec, "id", None),
            )
        except Exception as exc:
            # 永続化は best-effort。失敗してもAPI応答は維持（ログは残す）
            logging.exception("persist learn-event failed: %s", exc)

    # Kafkaへベストエフォート送信（有効時）
    try:
        logging.info("kafka_send_attempt event_id=%s", event.event_id)
        KafkaEventProducer().produce_json(event.model_dump())
        logging.info("kafka_send_completed event_id=%s", event.event_id)
    except Exception as e:
        logging.exception("kafka_send_failed event_id=%s: %s", event.event_id, e)

    return {"status": "accepted"}


@router.get("/stream/{user_id}")
async def stream_noncog_updates(user_id: str):
    """SSEストリーミングでリアルタイム更新を配信"""
    return await sse_manager.stream_events(user_id)


@router.get("/metrics/noncog-summary", response_model=NonCogSummary)
async def get_noncog_summary(
    user_id: str = Query(..., max_length=128),
) -> NonCogSummary:
    """ユーザーの非認知サマリ（インメモリ集計）を返す（MVP）"""
    agg = _USER_AGGREGATES.get(user_id)
    if not agg:
        raise HTTPException(status_code=404, detail="summary not found")

    grit, srl = _compute_scores(
        retry_count=agg["retry_count"],
        give_up_count=agg.get("give_up_count", 0),
        avg_think_time_ms=agg["avg_think_time_ms"],
    )

    total_hard = max(agg["retry_count"] + agg.get("give_up_count", 0), 1)
    re_rate = agg["retry_count"] / total_hard

    return NonCogSummary(
        user_id=user_id,
        retry_count=agg["retry_count"],
        avg_think_time_ms=agg["avg_think_time_ms"],
        re_challenge_rate=re_rate,
        grit_score=grit,
        srl_score=srl,
        updated_at=agg["updated_at"],
    )


@router.get("/metrics/noncog-summary-db", response_model=NonCogSummary)
async def get_noncog_summary_db(
    user_id: str = Query(..., max_length=128),
    db: AsyncSession = Depends(get_db),
) -> NonCogSummary:
    """DB集計に基づく非認知サマリ（MVP）。

    - retry_count: action_type = 'problem_retry'
    - give_up_count: action_type = 'give_up'
    - avg_think_time_ms: AVG(end_time - start_time)
    """
    try:
        numeric_user_id = int(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid user_id")

    from ...domain.models.behavior import BehaviorRecord  # local import to avoid cycles

    # counts
    q_retry = (
        select(func.count())
        .select_from(BehaviorRecord)
        .where(
            and_(
                BehaviorRecord.user_id == numeric_user_id,
                BehaviorRecord.action_type == "problem_retry",
            )
        )
    )
    q_giveup = (
        select(func.count())
        .select_from(BehaviorRecord)
        .where(
            and_(
                BehaviorRecord.user_id == numeric_user_id,
                BehaviorRecord.action_type == "give_up",
            )
        )
    )
    # avg think time (seconds) → ms
    q_durations = select(
        func.avg(
            func.extract("epoch", BehaviorRecord.end_time)
            - func.extract("epoch", BehaviorRecord.start_time)
        )
    ).where(
        and_(
            BehaviorRecord.user_id == numeric_user_id,
            BehaviorRecord.end_time.isnot(None),
            BehaviorRecord.start_time.isnot(None),
        )
    )

    retry_count = (await db.execute(q_retry)).scalar() or 0
    give_up_count = (await db.execute(q_giveup)).scalar() or 0
    avg_secs = (await db.execute(q_durations)).scalar()
    avg_ms = float(avg_secs) * 1000.0 if avg_secs is not None else 0.0

    total_hard = max(retry_count + give_up_count, 1)
    re_rate = retry_count / total_hard

    grit, srl = _compute_scores(
        retry_count=retry_count, give_up_count=give_up_count, avg_think_time_ms=avg_ms
    )

    return NonCogSummary(
        user_id=user_id,
        retry_count=retry_count,
        avg_think_time_ms=avg_ms,
        re_challenge_rate=re_rate,
        grit_score=grit,
        srl_score=srl,
        updated_at=datetime.utcnow(),
    )
