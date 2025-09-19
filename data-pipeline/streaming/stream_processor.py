from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json
import logging


class StreamProcessor:
    """リアルタイムデータ処理のためのSparkストリーミングプロセッサー"""

    def __init__(self, kafka_bootstrap_servers: str, kafka_topic: str):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.spark = self._create_spark_session()

    def _create_spark_session(self) -> SparkSession:
        """Sparkセッションの作成"""
        return (
            SparkSession.builder.appName("NonCog-StreamProcessor")
            .config("spark.streaming.stopGracefullyOnShutdown", "true")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )

    def _create_streaming_query(self, df):
        """ストリーミングクエリの作成と実行"""
        return df.writeStream.format("console").outputMode("append").start()

    def process_student_activity(self):
        """生徒のアクティビティデータの処理"""
        # スキーマ定義
        schema = StructType(
            [
                StructField("student_id", StringType(), True),
                StructField("activity_type", StringType(), True),
                StructField("timestamp", TimestampType(), True),
                StructField("data", StringType(), True),
            ]
        )

        # Kafkaからのストリーム読み込み
        df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
            .option("subscribe", self.kafka_topic)
            .load()
        )

        # データの変換
        parsed_df = df.select(
            from_json(col("value").cast("string"), schema).alias("parsed_data")
        ).select("parsed_data.*")

        # ウィンドウ集計
        windowed_counts = (
            parsed_df.withWatermark("timestamp", "10 minutes")
            .groupBy(window("timestamp", "5 minutes"), "student_id", "activity_type")
            .count()
        )

        # ストリーム処理の開始
        query = self._create_streaming_query(windowed_counts)
        return query

    def process_learning_metrics(self):
        """学習メトリクスの処理"""
        # スキーマ定義
        schema = StructType(
            [
                StructField("student_id", StringType(), True),
                StructField("metric_type", StringType(), True),
                StructField("value", DoubleType(), True),
                StructField("timestamp", TimestampType(), True),
            ]
        )

        # Kafkaからのストリーム読み込み
        df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
            .option("subscribe", self.kafka_topic)
            .load()
        )

        # データの変換と集計
        parsed_df = df.select(
            from_json(col("value").cast("string"), schema).alias("parsed_data")
        ).select("parsed_data.*")

        # 移動平均の計算
        metrics_ma = (
            parsed_df.withWatermark("timestamp", "10 minutes")
            .groupBy(window("timestamp", "5 minutes"), "student_id", "metric_type")
            .agg(avg("value").alias("moving_average"), stddev("value").alias("std_dev"))
        )

        # ストリーム処理の開始
        query = self._create_streaming_query(metrics_ma)
        return query


def main():
    """メイン実行関数"""
    logging.basicConfig(level=logging.INFO)

    # 設定
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    ACTIVITY_TOPIC = "student-activity"
    METRICS_TOPIC = "learning-metrics"

    # プロセッサーの初期化
    activity_processor = StreamProcessor(KAFKA_BOOTSTRAP_SERVERS, ACTIVITY_TOPIC)
    metrics_processor = StreamProcessor(KAFKA_BOOTSTRAP_SERVERS, METRICS_TOPIC)

    try:
        # ストリーム処理の開始
        activity_query = activity_processor.process_student_activity()
        metrics_query = metrics_processor.process_learning_metrics()

        # クエリの待機
        activity_query.awaitTermination()
        metrics_query.awaitTermination()

    except KeyboardInterrupt:
        logging.info("Shutting down stream processor...")
        activity_processor.spark.stop()
        metrics_processor.spark.stop()


if __name__ == "__main__":
    main()
