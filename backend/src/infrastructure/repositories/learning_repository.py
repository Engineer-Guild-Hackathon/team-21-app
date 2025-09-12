from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.models.behavior import ActionType, BehaviorRecord


class LearningEventRepository:
    """学習行動イベントを既存の BehaviorRecord に保存するための最小リポジトリ。

    現状は API 側のインメモリ集計を維持しつつ、将来 DB 永続化へ切り替える際の下準備として提供。
    """

    _ACTION_MAP: dict[str, ActionType] = {
        "answer_submit": ActionType.PROBLEM_SUBMIT,
        "hint_request": ActionType.HINT_REQUEST,
        "retry": ActionType.PROBLEM_RETRY,
        "give_up": ActionType.GIVE_UP,
    }

    @classmethod
    def map_action(cls, action: str) -> ActionType:
        if action not in cls._ACTION_MAP:
            raise ValueError(f"unsupported action: {action}")
        return cls._ACTION_MAP[action]

    async def create_from_learn_action(
        self,
        session: AsyncSession,
        *,
        user_id: int,
        action: str,
        think_time_ms: int,
        success: Optional[bool] = None,
        created_at: Optional[datetime] = None,
    ) -> BehaviorRecord:
        """LearnActionEvent から BehaviorRecord を生成し保存。

        problem_id など未確定の項目は MVP では None のままとする。
        start_time は created_at から think_time_ms を差し引いて擬似的に設定。
        """
        action_type = self.map_action(action)
        ended = created_at or datetime.utcnow()
        started = (
            ended
            if think_time_ms <= 0
            else ended - timedelta(milliseconds=think_time_ms)
        )

        record = BehaviorRecord(
            user_id=user_id,
            problem_id=None,
            action_type=(
                action_type.value if hasattr(action_type, "value") else str(action_type)
            ),
            start_time=started,
            end_time=ended,
            attempt_count=1,
            success=success,
            approach_description=None,
            emotion_state=None,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record
