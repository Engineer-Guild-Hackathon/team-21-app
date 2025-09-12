from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Set

from fastapi.responses import StreamingResponse


class SSEManager:
    """Server-Sent Events管理クラス"""

    def __init__(self) -> None:
        self._connections: Set[asyncio.Queue] = set()
        logging.info("sse_manager_initialized")

    def add_connection(self, queue: asyncio.Queue) -> None:
        """SSE接続を追加"""
        self._connections.add(queue)
        logging.info("sse_connection_added total=%d", len(self._connections))

    def remove_connection(self, queue: asyncio.Queue) -> None:
        """SSE接続を削除"""
        self._connections.discard(queue)
        logging.info("sse_connection_removed total=%d", len(self._connections))

    async def broadcast(self, data: Dict[str, Any]) -> None:
        """全接続にデータをブロードキャスト"""
        if not self._connections:
            return

        message = f"data: {json.dumps(data)}\n\n"
        dead_connections = set()

        for queue in self._connections:
            try:
                await queue.put(message)
            except Exception as e:
                logging.exception("sse_broadcast_error: %s", e)
                dead_connections.add(queue)

        # 死んだ接続を削除
        for dead_conn in dead_connections:
            self._connections.discard(dead_conn)

        logging.info("sse_broadcast_sent connections=%d", len(self._connections))

    async def stream_events(self, user_id: str) -> StreamingResponse:
        """SSEストリーミングレスポンスを生成"""
        queue = asyncio.Queue()
        self.add_connection(queue)

        async def event_generator():
            try:
                # 接続開始メッセージ
                yield f"data: {json.dumps({'type': 'connected', 'user_id': user_id})}\n\n"

                while True:
                    try:
                        # メッセージを待機（タイムアウト付き）
                        message = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield message
                    except asyncio.TimeoutError:
                        # キープアライブ
                        yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    except Exception as e:
                        logging.exception("sse_stream_error: %s", e)
                        break
            finally:
                self.remove_connection(queue)
                logging.info("sse_stream_ended user_id=%s", user_id)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )


# グローバルSSEマネージャー
sse_manager = SSEManager()
