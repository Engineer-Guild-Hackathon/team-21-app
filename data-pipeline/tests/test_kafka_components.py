import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime
import json
from ..kafka.kafka_producer import KafkaProducer, BatchKafkaProducer, RetryKafkaProducer
from ..kafka.kafka_consumer import (
    KafkaConsumer,
    BatchKafkaConsumer,
    ConsumerMessage,
    MessageType,
    MessageHandler,
)


class MockProducer:
    def __init__(self):
        self.produced_messages = []
        self.callbacks = []

    def produce(self, topic, value, callback=None):
        self.produced_messages.append((topic, value))
        if callback:
            self.callbacks.append(callback)

    def poll(self, timeout):
        return len(self.produced_messages)

    def flush(self, timeout):
        return 0

    def close(self):
        pass


class MockConsumer:
    def __init__(self):
        self.messages = []
        self.subscribed_topics = []
        self.position = 0

    def subscribe(self, topics):
        self.subscribed_topics = topics

    def poll(self, timeout):
        if self.position >= len(self.messages):
            return None
        message = self.messages[self.position]
        self.position += 1
        return message

    def close(self):
        pass

    def list_topics(self, topic):
        class MockPartition:
            high_watermark = 100

        class MockTopic:
            partitions = {0: MockPartition()}

        class MockMetadata:
            topics = {topic: MockTopic()}

        return MockMetadata()


@pytest.fixture
def mock_producer():
    return MockProducer()


@pytest.fixture
def mock_consumer():
    return MockConsumer()


@pytest.fixture
def kafka_producer():
    config = {"bootstrap.servers": "localhost:9092", "client.id": "test-producer"}
    with patch("confluent_kafka.Producer", return_value=MockProducer()):
        producer = KafkaProducer(config)
        yield producer


@pytest.fixture
def kafka_consumer():
    config = {
        "bootstrap.servers": "localhost:9092",
        "group.id": "test-consumer",
        "auto.offset.reset": "earliest",
    }
    topics = ["test-topic"]
    with patch("confluent_kafka.Consumer", return_value=MockConsumer()):
        consumer = KafkaConsumer(config, topics)
        yield consumer


class TestKafkaProducer:
    @pytest.mark.asyncio
    async def test_produce_student_activity(self, kafka_producer):
        """生徒のアクティビティデータ送信のテスト"""
        student_id = "test-student"
        activity_data = {
            "type": "problem_solved",
            "problem_id": "prob-123",
            "time_spent": 300,
        }

        await kafka_producer.produce_student_activity(student_id, activity_data)

        # プロデューサーの内部状態を確認
        produced_messages = kafka_producer.producer.produced_messages
        assert len(produced_messages) == 1

        topic, value = produced_messages[0]
        assert topic == kafka_producer.topics["student_activity"]

        message = json.loads(value.decode("utf-8"))
        assert message["student_id"] == student_id
        assert message["activity_type"] == activity_data["type"]

    @pytest.mark.asyncio
    async def test_produce_learning_metrics(self, kafka_producer):
        """学習メトリクス送信のテスト"""
        student_id = "test-student"
        metrics = {"accuracy": 0.85, "completion_rate": 0.75, "time_spent": 1200}

        await kafka_producer.produce_learning_metrics(student_id, metrics)

        produced_messages = kafka_producer.producer.produced_messages
        assert len(produced_messages) == 1

        topic, value = produced_messages[0]
        assert topic == kafka_producer.topics["learning_metrics"]

        message = json.loads(value.decode("utf-8"))
        assert message["student_id"] == student_id
        assert message["metrics"] == metrics

    @pytest.mark.asyncio
    async def test_batch_producer(self):
        """バッチプロデューサーのテスト"""
        config = {
            "bootstrap.servers": "localhost:9092",
            "client.id": "test-batch-producer",
        }

        with patch("confluent_kafka.Producer", return_value=MockProducer()):
            producer = BatchKafkaProducer(config, batch_size=2)

            # 2つのメッセージを送信
            await producer.produce_student_activity("student-1", {"type": "login"})
            await producer.produce_student_activity("student-2", {"type": "logout"})

            # バッチサイズに達したのでフラッシュされているはず
            assert producer.message_count == 0

    @pytest.mark.asyncio
    async def test_retry_producer(self):
        """リトライプロデューサーのテスト"""
        config = {
            "bootstrap.servers": "localhost:9092",
            "client.id": "test-retry-producer",
        }

        with patch("confluent_kafka.Producer", return_value=MockProducer()):
            producer = RetryKafkaProducer(config, max_retries=3, retry_interval=0.1)

            # エラーを発生させる
            with patch.object(
                producer,
                "_produce_message",
                side_effect=[Exception("Test error"), Exception("Test error"), None],
            ):
                await producer.produce_student_activity("student-1", {"type": "login"})


class TestKafkaConsumer:
    @pytest.mark.asyncio
    async def test_message_processing(self, kafka_consumer, mock_consumer):
        """メッセージ処理のテスト"""
        # テストメッセージの作成
        test_message = Mock()
        test_message.topic.return_value = MessageType.STUDENT_ACTIVITY.value
        test_message.partition.return_value = 0
        test_message.offset.return_value = 1
        test_message.key.return_value = None
        test_message.value.return_value = json.dumps(
            {"student_id": "test-student", "activity_type": "login"}
        ).encode("utf-8")
        test_message.timestamp.return_value = (
            1,
            int(datetime.now().timestamp() * 1000),
        )
        test_message.error.return_value = None

        # モックコンシューマーにメッセージを追加
        mock_consumer.messages.append(test_message)
        kafka_consumer.consumer = mock_consumer

        # 非同期処理のテスト用のタスク作成
        async def stop_consumer():
            await asyncio.sleep(0.1)
            kafka_consumer.stop()

        # コンシューマーの実行
        await asyncio.gather(kafka_consumer.start(), stop_consumer())

    @pytest.mark.asyncio
    async def test_batch_consumer(self):
        """バッチコンシューマーのテスト"""
        config = {
            "bootstrap.servers": "localhost:9092",
            "group.id": "test-batch-consumer",
            "auto.offset.reset": "earliest",
        }
        topics = ["test-topic"]

        with patch("confluent_kafka.Consumer", return_value=MockConsumer()):
            consumer = BatchKafkaConsumer(config, topics, batch_size=2)

            # テストメッセージの作成
            messages = []
            for i in range(3):
                msg = Mock()
                msg.topic.return_value = MessageType.STUDENT_ACTIVITY.value
                msg.partition.return_value = 0
                msg.offset.return_value = i
                msg.key.return_value = None
                msg.value.return_value = json.dumps(
                    {"student_id": f"student-{i}", "activity_type": "login"}
                ).encode("utf-8")
                msg.timestamp.return_value = (1, int(datetime.now().timestamp() * 1000))
                msg.error.return_value = None
                messages.append(msg)

            consumer.consumer.messages = messages

            # 非同期処理のテスト用のタスク作成
            async def stop_consumer():
                await asyncio.sleep(0.1)
                consumer.stop()

            # コンシューマーの実行
            await asyncio.gather(consumer.start(), stop_consumer())


class TestMessageHandlers:
    @pytest.mark.asyncio
    async def test_student_activity_handler(self):
        """生徒のアクティビティハンドラーのテスト"""
        handler = StudentActivityHandler()
        message = ConsumerMessage(
            topic=MessageType.STUDENT_ACTIVITY.value,
            partition=0,
            offset=1,
            key=None,
            value={
                "student_id": "test-student",
                "activity_type": "problem_solved",
                "problem_id": "prob-123",
            },
            timestamp=datetime.now(),
        )

        result = await handler.handle(message)
        assert result is True

    @pytest.mark.asyncio
    async def test_learning_metrics_handler(self):
        """学習メトリクスハンドラーのテスト"""
        handler = LearningMetricsHandler()
        message = ConsumerMessage(
            topic=MessageType.LEARNING_METRICS.value,
            partition=0,
            offset=1,
            key=None,
            value={
                "student_id": "test-student",
                "metrics": {"accuracy": 0.85, "completion_rate": 0.75},
            },
            timestamp=datetime.now(),
        )

        result = await handler.handle(message)
        assert result is True
