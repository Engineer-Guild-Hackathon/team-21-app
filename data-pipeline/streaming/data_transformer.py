from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from typing import Dict, List, Optional
import json
import logging
import numpy as np
from datetime import datetime, timedelta
import structlog


class DataTransformer:
    """データ変換・集計パイプライン"""

    def __init__(self, spark_session: Optional[SparkSession] = None):
        self.spark = spark_session or self._create_spark_session()
        self.logger = structlog.get_logger(__name__)

    def _create_spark_session(self) -> SparkSession:
        """Sparkセッションの作成"""
        return (
            SparkSession.builder.appName("NonCog-DataTransformer")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.memory.offHeap.enabled", "true")
            .config("spark.memory.offHeap.size", "2g")
            .getOrCreate()
        )

    def transform_student_activities(self, df):
        """生徒のアクティビティデータの変換"""
        # アクティビティの種類ごとの重み付け
        activity_weights = {
            "problem_solved": 1.0,
            "challenge_completed": 1.2,
            "help_requested": 0.5,
            "collaboration": 1.1,
            "review": 0.8,
        }

        # 重み付けUDF
        weight_udf = udf(
            lambda activity: float(activity_weights.get(activity, 0.0)), DoubleType()
        )

        # アクティビティスコアの計算
        activity_scores = df.withColumn(
            "activity_weight", weight_udf(col("activity_type"))
        ).withColumn(
            "activity_score",
            col("activity_weight")
            * when(col("success").isNotNull() & col("success"), 1.0).otherwise(0.5),
        )

        # 時間窓ごとの集計
        window_spec = (
            Window.partitionBy("student_id")
            .orderBy("timestamp")
            .rangeBetween(-timedelta(hours=1).total_seconds(), 0)
        )

        transformed_activities = (
            activity_scores.withColumn(
                "rolling_score", sum("activity_score").over(window_spec)
            )
            .withColumn("activity_count", count("activity_type").over(window_spec))
            .withColumn(
                "success_rate",
                avg(when(col("success"), 1.0).otherwise(0.0)).over(window_spec),
            )
            .withColumn(
                "activity_variety", size(collect_set("activity_type").over(window_spec))
            )
        )

        return transformed_activities

    def transform_learning_metrics(self, df):
        """学習メトリクスの変換"""
        # メトリクスの正規化
        metrics_stats = df.select(
            [
                mean(col(f"metrics.{metric}")).alias(f"{metric}_mean")
                for metric in ["accuracy", "completion_rate", "time_spent"]
            ]
        )

        normalized_metrics = df
        for metric in ["accuracy", "completion_rate", "time_spent"]:
            normalized_metrics = normalized_metrics.withColumn(
                f"normalized_{metric}",
                (col(f"metrics.{metric}") - col(f"{metric}_mean"))
                / stddev(col(f"metrics.{metric}")).over(Window.partitionBy()),
            )

        # 複合メトリクスの計算
        window_spec = (
            Window.partitionBy("student_id")
            .orderBy("timestamp")
            .rangeBetween(-timedelta(hours=1).total_seconds(), 0)
        )

        transformed_metrics = (
            normalized_metrics.withColumn(
                "learning_efficiency",
                col("normalized_accuracy")
                * col("normalized_completion_rate")
                / greatest(col("normalized_time_spent"), lit(0.1)),
            )
            .withColumn(
                "rolling_efficiency", avg("learning_efficiency").over(window_spec)
            )
            .withColumn(
                "improvement_rate",
                col("learning_efficiency")
                - lag("learning_efficiency", 1).over(
                    Window.partitionBy("student_id").orderBy("timestamp")
                ),
            )
        )

        return transformed_metrics

    def transform_emotional_states(self, df):
        """感情状態データの変換"""
        # 感情カテゴリのエンコーディング
        emotion_categories = [
            "motivated",
            "frustrated",
            "confused",
            "satisfied",
            "anxious",
            "confident",
            "bored",
        ]

        # One-hotエンコーディング
        encoded_emotions = df
        for emotion in emotion_categories:
            encoded_emotions = encoded_emotions.withColumn(
                f"is_{emotion}", when(col("emotion") == emotion, 1).otherwise(0)
            )

        # 感情状態の時系列特徴
        window_spec = (
            Window.partitionBy("student_id")
            .orderBy("timestamp")
            .rangeBetween(-timedelta(hours=1).total_seconds(), 0)
        )

        transformed_emotions = encoded_emotions
        for emotion in emotion_categories:
            transformed_emotions = transformed_emotions.withColumn(
                f"{emotion}_frequency",
                sum(col(f"is_{emotion}")).over(window_spec)
                / count("*").over(window_spec),
            )

        # 感情安定性スコアの計算
        transformed_emotions = transformed_emotions.withColumn(
            "emotional_stability",
            1.0
            - stddev_pop(array([col(f"{e}_frequency") for e in emotion_categories])),
        ).withColumn(
            "dominant_emotion",
            array_max(
                array(
                    [
                        struct(
                            col(f"{e}_frequency").alias("freq"), lit(e).alias("emotion")
                        )
                        for e in emotion_categories
                    ]
                )
            ).emotion,
        )

        return transformed_emotions

    def create_integrated_features(self, activities_df, metrics_df, emotions_df):
        """統合特徴量の作成"""
        # 時間窓の定義
        window_spec = (
            Window.partitionBy("student_id")
            .orderBy("timestamp")
            .rangeBetween(-timedelta(hours=1).total_seconds(), 0)
        )

        # アクティビティ特徴量
        activity_features = activities_df.select(
            "student_id",
            "timestamp",
            "rolling_score",
            "activity_count",
            "success_rate",
            "activity_variety",
        )

        # 学習メトリクス特徴量
        metrics_features = metrics_df.select(
            "student_id",
            "timestamp",
            "learning_efficiency",
            "rolling_efficiency",
            "improvement_rate",
        )

        # 感情特徴量
        emotion_features = emotions_df.select(
            "student_id",
            "timestamp",
            "emotional_stability",
            "dominant_emotion",
            *[
                f"{e}_frequency"
                for e in [
                    "motivated",
                    "frustrated",
                    "confused",
                    "satisfied",
                    "anxious",
                    "confident",
                    "bored",
                ]
            ],
        )

        # 特徴量の結合
        joined_features = activity_features.join(
            metrics_features, ["student_id", "timestamp"], "outer"
        ).join(emotion_features, ["student_id", "timestamp"], "outer")

        # 欠損値の補完
        filled_features = joined_features.na.fill(
            {
                "rolling_score": 0,
                "activity_count": 0,
                "success_rate": 0.5,
                "activity_variety": 0,
                "learning_efficiency": 0,
                "rolling_efficiency": 0,
                "improvement_rate": 0,
                "emotional_stability": 0.5,
            }
        )

        # 複合特徴量の計算
        integrated_features = (
            filled_features.withColumn(
                "engagement_score",
                (
                    col("rolling_score") * 0.3
                    + col("success_rate") * 0.3
                    + col("emotional_stability") * 0.4
                ),
            )
            .withColumn(
                "learning_potential",
                (
                    col("learning_efficiency") * 0.4
                    + col("improvement_rate") * 0.3
                    + col("activity_variety") * 0.3
                ),
            )
            .withColumn(
                "emotional_balance",
                (
                    col("emotional_stability") * 0.5
                    + when(
                        col("dominant_emotion").isin(
                            ["motivated", "confident", "satisfied"]
                        ),
                        1.0,
                    ).otherwise(0.0)
                    * 0.5
                ),
            )
        )

        return integrated_features

    def aggregate_student_metrics(self, integrated_features):
        """生徒ごとのメトリクス集計"""
        # 時間窓の定義
        daily_window = (
            Window.partitionBy("student_id")
            .orderBy("timestamp")
            .rangeBetween(-timedelta(days=1).total_seconds(), 0)
        )

        weekly_window = (
            Window.partitionBy("student_id")
            .orderBy("timestamp")
            .rangeBetween(-timedelta(weeks=1).total_seconds(), 0)
        )

        # 集計の実行
        aggregated_metrics = (
            integrated_features.withColumn(
                "daily_engagement", avg("engagement_score").over(daily_window)
            )
            .withColumn("daily_learning", avg("learning_potential").over(daily_window))
            .withColumn("daily_emotional", avg("emotional_balance").over(daily_window))
            .withColumn(
                "weekly_engagement", avg("engagement_score").over(weekly_window)
            )
            .withColumn(
                "weekly_learning", avg("learning_potential").over(weekly_window)
            )
            .withColumn(
                "weekly_emotional", avg("emotional_balance").over(weekly_window)
            )
            .withColumn(
                "engagement_trend", col("daily_engagement") - col("weekly_engagement")
            )
            .withColumn(
                "learning_trend", col("daily_learning") - col("weekly_learning")
            )
            .withColumn(
                "emotional_trend", col("daily_emotional") - col("weekly_emotional")
            )
        )

        return aggregated_metrics

    def save_transformed_data(self, df, output_path: str, mode: str = "append"):
        """変換したデータの保存"""
        try:
            df.write.mode(mode).parquet(output_path)

            self.logger.info("Data saved successfully", path=output_path, mode=mode)

        except Exception as e:
            self.logger.error("Failed to save data", path=output_path, error=str(e))
            raise

    def process_streaming_data(self, input_stream_df, output_path: str):
        """ストリーミングデータの処理"""

        def process_batch(batch_df, batch_id):
            try:
                # バッチデータの変換
                if "activity_type" in batch_df.columns:
                    transformed_df = self.transform_student_activities(batch_df)
                elif "metrics" in batch_df.columns:
                    transformed_df = self.transform_learning_metrics(batch_df)
                elif "emotion" in batch_df.columns:
                    transformed_df = self.transform_emotional_states(batch_df)
                else:
                    self.logger.warning("Unknown data type in batch", batch_id=batch_id)
                    return

                # 変換データの保存
                self.save_transformed_data(
                    transformed_df, f"{output_path}/batch_{batch_id}", "overwrite"
                )

                self.logger.info("Batch processed successfully", batch_id=batch_id)

            except Exception as e:
                self.logger.error(
                    "Failed to process batch", batch_id=batch_id, error=str(e)
                )
                raise

        # ストリーミング処理の開始
        return (
            input_stream_df.writeStream.foreachBatch(process_batch)
            .outputMode("append")
            .start()
        )
