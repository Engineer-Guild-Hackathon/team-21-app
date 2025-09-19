from confluent_kafka import Consumer, KafkaError
import json
import logging
from typing import Dict, Any, Callable, List, Optional
from datetime import datetime
import asyncio
from prometheus_client import Counter, Histogram, Gauge
import structlog
from dataclasses import dataclass
from enum import Enum

# メトリクスの定義
CONSUMED_MESSAGES = Counter(
    "kafka_consumer_messages_total",
    "Total number of messages consumed",
    ["topic", "status"],
)

PROCESSING_TIME = Histogram(
    "kafka_consumer_processing_seconds", "Time taken to process messages", ["topic"]
)

CONSUMER_LAG = Gauge(
    "kafka_consumer_lag", "Consumer lag in messages", ["topic", "partition"]
)


class MessageType(Enum):
    """メッセージタイプの定義"""

    STUDENT_ACTIVITY = "student_activity"
    LEARNING_METRICS = "learning_metrics"
    EMOTIONAL_STATE = "emotional_state"
    FEEDBACK_EVENTS = "feedback_events"


@dataclass
class ConsumerMessage:
    """消費されたメッセージを表すデータクラス"""

    topic: str
    partition: int
    offset: int
    key: Optional[str]
    value: Dict[str, Any]
    timestamp: datetime


class MessageHandler:
    """メッセージ処理の基底クラス"""

    async def handle(self, message: ConsumerMessage) -> bool:
        """メッセージの処理"""
        raise NotImplementedError


class StudentActivityHandler(MessageHandler):
    """生徒のアクティビティメッセージの処理"""

    async def handle(self, message: ConsumerMessage) -> bool:
        # アクティビティデータの処理ロジック
        try:
            activity_data = message.value
            # ここで具体的な処理を実装
            return True
        except Exception as e:
            logging.error(f"Failed to process student activity: {str(e)}")
            return False


class LearningMetricsHandler(MessageHandler):
    """学習メトリクスメッセージの処理"""

    async def handle(self, message: ConsumerMessage) -> bool:
        # メトリクスデータの処理ロジック
        try:
            metrics_data = message.value
            # ここで具体的な処理を実装
            return True
        except Exception as e:
            logging.error(f"Failed to process learning metrics: {str(e)}")
            return False


class EmotionalStateHandler(MessageHandler):
    """感情状態メッセージの処理"""

    async def handle(self, message: ConsumerMessage) -> bool:
        # 感情状態データの処理ロジック
        try:
            emotional_data = message.value
            # ここで具体的な処理を実装
            return True
        except Exception as e:
            logging.error(f"Failed to process emotional state: {str(e)}")
            return False


class FeedbackEventHandler(MessageHandler):
    """フィードバックイベントメッセージの処理"""

    async def handle(self, message: ConsumerMessage) -> bool:
        # フィードバックデータの処理ロジック
        try:
            feedback_data = message.value
            # ここで具体的な処理を実装
            return True
        except Exception as e:
            logging.error(f"Failed to process feedback event: {str(e)}")
            return False


class KafkaConsumer:
    """非認知学習プラットフォームのKafkaコンシューマー"""

    def __init__(self, config: Dict[str, Any], topics: List[str]):
        self.config = config
        self.consumer = Consumer(config)
        self.topics = topics
        self.running = False

        # ロガーの設定
        self.logger = structlog.get_logger(__name__)

        # メッセージハンドラーの設定
        self.handlers = {
            MessageType.STUDENT_ACTIVITY.value: StudentActivityHandler(),
            MessageType.LEARNING_METRICS.value: LearningMetricsHandler(),
            MessageType.EMOTIONAL_STATE.value: EmotionalStateHandler(),
            MessageType.FEEDBACK_EVENTS.value: FeedbackEventHandler(),
        }

    async def start(self):
        """コンシューマーの開始"""
        try:
            self.consumer.subscribe(self.topics)
            self.running = True

            self.logger.info("Consumer started", topics=self.topics)

            while self.running:
                try:
                    # メッセージの取得
                    msg = self.consumer.poll(timeout=1.0)

                    if msg is None:
                        continue

                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            self.logger.info(
                                "Reached end of partition",
                                topic=msg.topic(),
                                partition=msg.partition(),
                            )
                        else:
                            self.logger.error("Consumer error", error=str(msg.error()))
                        continue

                    # メッセージの処理
                    await self._process_message(msg)

                except Exception as e:
                    self.logger.error("Error processing message", error=str(e))

        finally:
            self.consumer.close()

    async def _process_message(self, msg):
        """メッセージの処理"""
        try:
            # メッセージの解析
            value = json.loads(msg.value().decode("utf-8"))

            message = ConsumerMessage(
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
                key=msg.key().decode("utf-8") if msg.key() else None,
                value=value,
                timestamp=datetime.fromtimestamp(msg.timestamp()[1] / 1000),
            )

            # メトリクスの記録開始
            with PROCESSING_TIME.labels(topic=message.topic).time():
                # メッセージタイプに応じたハンドラーの取得と実行
                handler = self.handlers.get(message.topic)
                if handler:
                    success = await handler.handle(message)
                    if success:
                        CONSUMED_MESSAGES.labels(
                            topic=message.topic, status="success"
                        ).inc()
                    else:
                        CONSUMED_MESSAGES.labels(
                            topic=message.topic, status="failed"
                        ).inc()
                else:
                    self.logger.warning(
                        "No handler found for topic", topic=message.topic
                    )

            # コンシューマーラグの更新
            self._update_consumer_lag(message)

        except json.JSONDecodeError as e:
            self.logger.error(
                "Failed to decode message", topic=msg.topic(), error=str(e)
            )
            CONSUMED_MESSAGES.labels(topic=msg.topic(), status="failed").inc()

        except Exception as e:
            self.logger.error(
                "Failed to process message", topic=msg.topic(), error=str(e)
            )
            CONSUMED_MESSAGES.labels(topic=msg.topic(), status="failed").inc()

    def _update_consumer_lag(self, message: ConsumerMessage):
        """コンシューマーラグの更新"""
        # 現在のオフセットと最新オフセットの差を計算
        metadata = self.consumer.list_topics(message.topic)
        partition = metadata.topics[message.topic].partitions[message.partition]
        current_offset = message.offset
        last_offset = partition.high_watermark

        lag = last_offset - current_offset
        CONSUMER_LAG.labels(topic=message.topic, partition=message.partition).set(lag)

    def stop(self):
        """コンシューマーの停止"""
        self.running = False


class BatchKafkaConsumer(KafkaConsumer):
    """バッチ処理用のKafkaコンシューマー"""

    def __init__(
        self, config: Dict[str, Any], topics: List[str], batch_size: int = 100
    ):
        super().__init__(config, topics)
        self.batch_size = batch_size

    async def start(self):
        """バッチ処理モードでのコンシューマーの開始"""
        try:
            self.consumer.subscribe(self.topics)
            self.running = True

            while self.running:
                messages = []

                # バッチサイズ分のメッセージを収集
                for _ in range(self.batch_size):
                    msg = self.consumer.poll(timeout=1.0)
                    if msg is None:
                        break

                    if msg.error():
                        if msg.error().code() != KafkaError._PARTITION_EOF:
                            self.logger.error("Consumer error", error=str(msg.error()))
                        continue

                    messages.append(msg)

                if messages:
                    # バッチ処理の実行
                    await self._process_batch(messages)

        finally:
            self.consumer.close()

    async def _process_batch(self, messages: List[Any]):
        """メッセージバッチの処理"""
        for msg in messages:
            await self._process_message(msg)
