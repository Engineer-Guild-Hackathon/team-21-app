from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from confluent_kafka import Producer


class KafkaEventProducer:
    """Thin wrapper for confluent_kafka. Best-effort fire-and-forget producer.

    Controlled by ENABLE_KAFKA_PRODUCE env var. Safe to construct even if disabled.
    """

    def __init__(self, bootstrap_servers: Optional[str] = None) -> None:
        self.enabled = os.getenv("ENABLE_KAFKA_PRODUCE", "false").lower() == "true"
        self.topic = os.getenv("KAFKA_LEARN_EVENT_TOPIC", "learn_action_events")
        servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"
        )
        logging.info(
            "kafka_producer_init enabled=%s topic=%s servers=%s",
            self.enabled,
            self.topic,
            servers,
        )
        self._producer: Optional[Producer] = None
        if self.enabled:
            try:
                self._producer = Producer(
                    {
                        "bootstrap.servers": servers,
                        "acks": "all",
                        "enable.idempotence": True,
                        "linger.ms": 0,
                    }
                )
                logging.info("kafka_producer_created successfully")
            except Exception as e:
                logging.exception("kafka_producer_creation_failed: %s", e)
                self.enabled = False

    def produce_json(self, payload: dict[str, Any]) -> None:
        if not self.enabled or not self._producer:
            logging.info("kafka_producer_disabled or not initialized")
            return
        try:
            logging.info(
                "kafka_produce_attempt topic=%s payload_size=%d",
                self.topic,
                len(str(payload)),
            )

            def _delivery(err, msg):
                if err is not None:
                    logging.error("kafka_delivery_failed: %s", err)
                else:
                    logging.info(
                        "kafka_delivery_ok topic=%s partition=%s offset=%s",
                        msg.topic(),
                        msg.partition(),
                        msg.offset(),
                    )

            # JSONシリアライゼーション時にdatetimeオブジェクトを文字列に変換
            def json_serializer(obj):
                if hasattr(obj, "isoformat"):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            self._producer.produce(
                self.topic,
                json.dumps(payload, default=json_serializer).encode("utf-8"),
                callback=_delivery,
            )
            logging.info("kafka_produce_message_sent topic=%s", self.topic)

            # drive delivery callbacks and ensure immediate delivery for demo
            self._producer.poll(0.5)
            try:
                remaining = self._producer.flush(5.0)
                if remaining > 0:
                    logging.warning("kafka_flush_incomplete remaining=%d", remaining)
                else:
                    logging.info("kafka_flush_completed successfully")
            except Exception:
                logging.exception("kafka_flush_error")
        except Exception as e:
            # best-effort: swallow errors
            logging.exception("kafka_produce_failed: %s", str(e))

    def flush(self, timeout: float = 0.5) -> None:
        if not self.enabled or not self._producer:
            return
        try:
            self._producer.flush(timeout)
        except Exception:
            pass
