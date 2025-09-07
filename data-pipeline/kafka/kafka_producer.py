from confluent_kafka import Producer
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
from prometheus_client import Counter, Histogram
import structlog

# メトリクスの定義
PRODUCED_MESSAGES = Counter(
    'kafka_producer_messages_total',
    'Total number of messages produced',
    ['topic', 'status']
)

MESSAGE_SIZE = Histogram(
    'kafka_producer_message_size_bytes',
    'Size of produced messages in bytes',
    ['topic']
)

PRODUCE_LATENCY = Histogram(
    'kafka_producer_latency_seconds',
    'Latency of message production',
    ['topic']
)

class KafkaProducer:
    """非認知学習プラットフォームのKafkaプロデューサー"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.producer = Producer(config)
        
        # ロガーの設定
        self.logger = structlog.get_logger(__name__)
        
        # トピックの定義
        self.topics = {
            'student_activity': 'noncog.student.activity',
            'learning_metrics': 'noncog.learning.metrics',
            'emotional_state': 'noncog.emotional.state',
            'feedback_events': 'noncog.feedback.events'
        }
    
    def delivery_callback(self, err: Optional[Exception], msg: Any):
        """メッセージ配信のコールバック"""
        if err:
            self.logger.error("Message delivery failed", error=str(err))
            PRODUCED_MESSAGES.labels(topic=msg.topic(), status='failed').inc()
        else:
            self.logger.info("Message delivered", 
                           topic=msg.topic(),
                           partition=msg.partition(),
                           offset=msg.offset())
            PRODUCED_MESSAGES.labels(topic=msg.topic(), status='success').inc()
    
    async def produce_student_activity(self, student_id: str, activity_data: Dict[str, Any]):
        """生徒のアクティビティデータの送信"""
        message = {
            'student_id': student_id,
            'timestamp': datetime.utcnow().isoformat(),
            'activity_type': activity_data.get('type'),
            'data': activity_data
        }
        
        await self._produce_message(self.topics['student_activity'], message)
    
    async def produce_learning_metrics(self, student_id: str, metrics: Dict[str, float]):
        """学習メトリクスの送信"""
        message = {
            'student_id': student_id,
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics
        }
        
        await self._produce_message(self.topics['learning_metrics'], message)
    
    async def produce_emotional_state(self, student_id: str,
                                    emotional_state: Dict[str, Any]):
        """感情状態の送信"""
        message = {
            'student_id': student_id,
            'timestamp': datetime.utcnow().isoformat(),
            'emotional_state': emotional_state
        }
        
        await self._produce_message(self.topics['emotional_state'], message)
    
    async def produce_feedback_event(self, student_id: str,
                                   feedback_data: Dict[str, Any]):
        """フィードバックイベントの送信"""
        message = {
            'student_id': student_id,
            'timestamp': datetime.utcnow().isoformat(),
            'feedback': feedback_data
        }
        
        await self._produce_message(self.topics['feedback_events'], message)
    
    async def _produce_message(self, topic: str, message: Dict[str, Any]):
        """メッセージの送信"""
        try:
            # メッセージのシリアライズ
            value = json.dumps(message).encode('utf-8')
            
            # メトリクスの記録
            MESSAGE_SIZE.labels(topic=topic).observe(len(value))
            
            with PRODUCE_LATENCY.labels(topic=topic).time():
                # メッセージの送信
                self.producer.produce(
                    topic=topic,
                    value=value,
                    callback=self.delivery_callback
                )
                
                # バッファのフラッシュ
                self.producer.poll(0)
            
            self.logger.info("Message produced",
                           topic=topic,
                           message_size=len(value))
            
        except Exception as e:
            self.logger.error("Failed to produce message",
                            topic=topic,
                            error=str(e))
            raise
    
    async def flush(self, timeout: float = 10.0):
        """残りのメッセージをフラッシュ"""
        remaining = self.producer.flush(timeout)
        if remaining > 0:
            self.logger.warning(f"{remaining} messages remain unflushed")
        return remaining
    
    def close(self):
        """プロデューサーのクローズ"""
        try:
            remaining = self.producer.flush(5.0)
            if remaining > 0:
                self.logger.warning(f"{remaining} messages remain unflushed on close")
        finally:
            self.producer.close()

class BatchKafkaProducer(KafkaProducer):
    """バッチ処理用のKafkaプロデューサー"""
    
    def __init__(self, config: Dict[str, Any], batch_size: int = 100,
                 flush_interval: float = 1.0):
        super().__init__(config)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.message_count = 0
        self.last_flush_time = datetime.utcnow()
    
    async def _produce_message(self, topic: str, message: Dict[str, Any]):
        """バッチ処理を考慮したメッセージの送信"""
        await super()._produce_message(topic, message)
        
        self.message_count += 1
        now = datetime.utcnow()
        
        # バッチサイズに達したか、フラッシュ間隔を超えた場合にフラッシュ
        if (self.message_count >= self.batch_size or
            (now - self.last_flush_time).total_seconds() >= self.flush_interval):
            await self.flush()
            self.message_count = 0
            self.last_flush_time = now

class RetryKafkaProducer(KafkaProducer):
    """リトライ機能付きのKafkaプロデューサー"""
    
    def __init__(self, config: Dict[str, Any], max_retries: int = 3,
                 retry_interval: float = 1.0):
        super().__init__(config)
        self.max_retries = max_retries
        self.retry_interval = retry_interval
    
    async def _produce_message(self, topic: str, message: Dict[str, Any]):
        """リトライ機能付きのメッセージ送信"""
        for attempt in range(self.max_retries):
            try:
                await super()._produce_message(topic, message)
                return
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                
                self.logger.warning("Retrying message production",
                                  topic=topic,
                                  attempt=attempt + 1,
                                  error=str(e))
                
                await asyncio.sleep(self.retry_interval * (attempt + 1))
