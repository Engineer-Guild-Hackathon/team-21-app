from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class AdvancedStreamProcessor:
    """拡張Sparkストリーミングプロセッサー"""

    def __init__(self, kafka_bootstrap_servers: str):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.spark = self._create_spark_session()

        # スキーマの定義
        self.schemas = self._define_schemas()

        # ロガーの設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _create_spark_session(self) -> SparkSession:
        """Sparkセッションの作成"""
        return (
            SparkSession.builder.appName("NonCog-AdvancedStreamProcessor")
            .config("spark.streaming.stopGracefullyOnShutdown", "true")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoint")
            .config("spark.memory.offHeap.enabled", "true")
            .config("spark.memory.offHeap.size", "1g")
            .getOrCreate()
        )

    def _define_schemas(self) -> Dict[str, StructType]:
        """各種データスキーマの定義"""
        return {
            "student_activity": StructType(
                [
                    StructField("student_id", StringType(), True),
                    StructField("timestamp", TimestampType(), True),
                    StructField("activity_type", StringType(), True),
                    StructField("data", StringType(), True),
                ]
            ),
            "learning_metrics": StructType(
                [
                    StructField("student_id", StringType(), True),
                    StructField("timestamp", TimestampType(), True),
                    StructField(
                        "metrics",
                        StructType(
                            [
                                StructField("accuracy", DoubleType(), True),
                                StructField("completion_rate", DoubleType(), True),
                                StructField("time_spent", DoubleType(), True),
                                StructField("help_requests", IntegerType(), True),
                            ]
                        ),
                    ),
                ]
            ),
            "emotional_state": StructType(
                [
                    StructField("student_id", StringType(), True),
                    StructField("timestamp", TimestampType(), True),
                    StructField("emotion", StringType(), True),
                    StructField("confidence", DoubleType(), True),
                    StructField("features", ArrayType(DoubleType()), True),
                ]
            ),
        }

    def process_student_behavior_patterns(self):
        """生徒の行動パターン分析"""
        # 活動データの読み込み
        activity_df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
            .option("subscribe", "noncog.student.activity")
            .load()
        )

        # データの解析
        parsed_activity = activity_df.select(
            from_json(
                col("value").cast("string"), self.schemas["student_activity"]
            ).alias("parsed_data")
        ).select("parsed_data.*")

        # 行動パターンの特徴抽出
        behavior_features = (
            parsed_activity.withWatermark("timestamp", "1 hour")
            .groupBy(window("timestamp", "30 minutes"), "student_id")
            .agg(
                count("activity_type").alias("activity_count"),
                collect_list("activity_type").alias("activity_sequence"),
                avg(
                    when(col("activity_type") == "problem_solved", 1).otherwise(0)
                ).alias("problem_solving_rate"),
            )
        )

        # クラスタリングの準備
        assembler = VectorAssembler(
            inputCols=["activity_count", "problem_solving_rate"], outputCol="features"
        )

        behavior_vectors = assembler.transform(behavior_features)

        # クラスタリングモデルの適用
        kmeans = KMeans(k=3, featuresCol="features")
        behavior_clusters = kmeans.transform(behavior_vectors)

        # 結果の出力
        return (
            behavior_clusters.writeStream.outputMode("append").format("console").start()
        )

    def analyze_learning_progression(self):
        """学習進捗の分析"""
        # メトリクスデータの読み込み
        metrics_df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
            .option("subscribe", "noncog.learning.metrics")
            .load()
        )

        # データの解析
        parsed_metrics = metrics_df.select(
            from_json(
                col("value").cast("string"), self.schemas["learning_metrics"]
            ).alias("parsed_data")
        ).select("parsed_data.*")

        # 学習進捗の分析
        learning_progress = (
            parsed_metrics.withWatermark("timestamp", "1 hour")
            .groupBy(window("timestamp", "1 hour"), "student_id")
            .agg(
                avg("metrics.accuracy").alias("avg_accuracy"),
                avg("metrics.completion_rate").alias("avg_completion_rate"),
                sum("metrics.time_spent").alias("total_time_spent"),
                sum("metrics.help_requests").alias("total_help_requests"),
            )
        )

        # 進捗予測モデルの準備
        assembler = VectorAssembler(
            inputCols=[
                "avg_accuracy",
                "avg_completion_rate",
                "total_time_spent",
                "total_help_requests",
            ],
            outputCol="features",
        )

        progress_vectors = assembler.transform(learning_progress)

        # ランダムフォレストモデルの適用
        rf = RandomForestClassifier(
            labelCol="prediction_label", featuresCol="features", numTrees=10
        )

        progress_predictions = rf.transform(progress_vectors)

        # 結果の出力
        return (
            progress_predictions.writeStream.outputMode("append")
            .format("console")
            .start()
        )

    def detect_emotional_patterns(self):
        """感情パターンの検出"""
        # 感情データの読み込み
        emotion_df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
            .option("subscribe", "noncog.emotional.state")
            .load()
        )

        # データの解析
        parsed_emotions = emotion_df.select(
            from_json(
                col("value").cast("string"), self.schemas["emotional_state"]
            ).alias("parsed_data")
        ).select("parsed_data.*")

        # 感情パターンの分析
        emotion_patterns = (
            parsed_emotions.withWatermark("timestamp", "30 minutes")
            .groupBy(window("timestamp", "15 minutes"), "student_id")
            .agg(
                collect_list("emotion").alias("emotion_sequence"),
                avg("confidence").alias("avg_confidence"),
                collect_list("features").alias("feature_sequence"),
            )
        )

        # 感情遷移の分析
        emotion_transitions = emotion_patterns.select(
            "student_id",
            "window",
            explode(
                arrays_zip(
                    slice(col("emotion_sequence"), 1, -1),
                    slice(col("emotion_sequence"), 2, 0),
                )
            ).alias("transition"),
        ).select(
            "student_id",
            "window",
            col("transition.0").alias("from_emotion"),
            col("transition.1").alias("to_emotion"),
        )

        # 結果の出力
        return (
            emotion_transitions.writeStream.outputMode("append")
            .format("console")
            .start()
        )

    def analyze_integrated_learning_experience(self):
        """統合学習体験の分析"""
        # 各種データストリームの読み込み
        activity_df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
            .option("subscribe", "noncog.student.activity")
            .load()
        )

        metrics_df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
            .option("subscribe", "noncog.learning.metrics")
            .load()
        )

        emotion_df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers)
            .option("subscribe", "noncog.emotional.state")
            .load()
        )

        # データの解析と結合
        parsed_activity = activity_df.select(
            from_json(
                col("value").cast("string"), self.schemas["student_activity"]
            ).alias("activity_data")
        ).select("activity_data.*")

        parsed_metrics = metrics_df.select(
            from_json(
                col("value").cast("string"), self.schemas["learning_metrics"]
            ).alias("metrics_data")
        ).select("metrics_data.*")

        parsed_emotions = emotion_df.select(
            from_json(
                col("value").cast("string"), self.schemas["emotional_state"]
            ).alias("emotion_data")
        ).select("emotion_data.*")

        # ウィンドウ結合
        integrated_data = (
            parsed_activity.withWatermark("timestamp", "1 hour")
            .join(
                parsed_metrics.withWatermark("timestamp", "1 hour"),
                expr(
                    """
                    student_id = metrics_data.student_id AND
                    timestamp >= metrics_data.timestamp AND
                    timestamp <= metrics_data.timestamp + INTERVAL 5 MINUTES
                """
                ),
                "left_outer",
            )
            .join(
                parsed_emotions.withWatermark("timestamp", "1 hour"),
                expr(
                    """
                    student_id = emotion_data.student_id AND
                    timestamp >= emotion_data.timestamp AND
                    timestamp <= emotion_data.timestamp + INTERVAL 5 MINUTES
                """
                ),
                "left_outer",
            )
        )

        # 統合分析
        learning_experience = integrated_data.groupBy(
            window("timestamp", "15 minutes"), "student_id"
        ).agg(
            collect_list("activity_type").alias("activities"),
            avg("metrics.accuracy").alias("avg_accuracy"),
            avg("metrics.completion_rate").alias("avg_completion_rate"),
            collect_list("emotion").alias("emotions"),
            avg("confidence").alias("avg_emotional_confidence"),
        )

        # 結果の出力
        return (
            learning_experience.writeStream.outputMode("append")
            .format("console")
            .start()
        )

    def start_all_processors(self):
        """全プロセッサーの起動"""
        queries = []

        try:
            # 行動パターン分析の開始
            behavior_query = self.process_student_behavior_patterns()
            queries.append(behavior_query)

            # 学習進捗分析の開始
            progress_query = self.analyze_learning_progression()
            queries.append(progress_query)

            # 感情パターン分析の開始
            emotion_query = self.detect_emotional_patterns()
            queries.append(emotion_query)

            # 統合分析の開始
            integrated_query = self.analyze_integrated_learning_experience()
            queries.append(integrated_query)

            # 全クエリの待機
            for query in queries:
                query.awaitTermination()

        except Exception as e:
            self.logger.error(f"Error in stream processing: {str(e)}")
            for query in queries:
                query.stop()
            raise

        finally:
            self.spark.stop()
