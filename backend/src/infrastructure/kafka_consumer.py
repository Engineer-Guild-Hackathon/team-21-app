from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, Optional

from confluent_kafka import Consumer, KafkaError


class KafkaEventConsumer:
    """Kafkaコンシューマでリアルタイム集計とクライアント通知を実装"""

    def __init__(self, bootstrap_servers: Optional[str] = None) -> None:
        self.enabled = os.getenv("ENABLE_KAFKA_CONSUME", "true").lower() == "true"
        self.topic = os.getenv("KAFKA_LEARN_EVENT_TOPIC", "learn_action_events")
        servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"
        )

        logging.info(
            "kafka_consumer_init enabled=%s topic=%s servers=%s",
            self.enabled,
            self.topic,
            servers,
        )

        self._consumer: Optional[Consumer] = None
        self._running = False
        self._event_handlers: list[Callable[[Dict[str, Any]], None]] = []

        if self.enabled:
            try:
                self._consumer = Consumer(
                    {
                        "bootstrap.servers": servers,
                        "group.id": "noncog-backend-consumer",
                        "auto.offset.reset": "latest",
                        "enable.auto.commit": True,
                    }
                )
                logging.info("kafka_consumer_created successfully")
            except Exception as e:
                logging.exception("kafka_consumer_creation_failed: %s", e)
                self.enabled = False

    def add_event_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """イベント受信時のハンドラを追加"""
        self._event_handlers.append(handler)
        logging.info("kafka_consumer_handler_added count=%d", len(self._event_handlers))

    async def start_consuming(self) -> None:
        """Kafkaコンシューマを開始（非同期）"""
        if not self.enabled or not self._consumer:
            logging.warning("kafka_consumer_disabled, skipping start")
            return

        try:
            self._consumer.subscribe([self.topic])
            self._running = True
            logging.info("kafka_consumer_started topic=%s", self.topic)

            while self._running:
                try:
                    # 非同期でポーリング
                    msg = await asyncio.to_thread(self._consumer.poll, 1.0)

                    if msg is None:
                        continue

                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        else:
                            logging.error("kafka_consumer_error: %s", msg.error())
                            continue

                    # メッセージをパースしてハンドラに通知
                    try:
                        payload = json.loads(msg.value().decode("utf-8"))
                        logging.info(
                            "kafka_consumer_message_received topic=%s partition=%s offset=%s",
                            msg.topic(),
                            msg.partition(),
                            msg.offset(),
                        )

                        # 全ハンドラに通知
                        for handler in self._event_handlers:
                            try:
                                handler(payload)
                            except Exception as e:
                                logging.exception("kafka_consumer_handler_error: %s", e)

                    except Exception as e:
                        logging.exception("kafka_consumer_parse_error: %s", e)

                except Exception as e:
                    logging.exception("kafka_consumer_poll_error: %s", e)
                    await asyncio.sleep(1.0)

        except Exception as e:
            logging.exception("kafka_consumer_start_error: %s", e)
        finally:
            if self._consumer:
                self._consumer.close()
            logging.info("kafka_consumer_stopped")

    def stop_consuming(self) -> None:
        """Kafkaコンシューマを停止"""
        self._running = False
        logging.info("kafka_consumer_stop_requested")
