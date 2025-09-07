from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.stat import Correlation
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from prometheus_client import Counter, Gauge, Histogram
import json

# メトリクスの定義
ANOMALY_COUNTER = Counter(
    'noncog_anomalies_total',
    'Total number of detected anomalies',
    ['type', 'severity']
)

SYSTEM_HEALTH = Gauge(
    'noncog_system_health',
    'Overall system health score',
    ['component']
)

PROCESSING_LATENCY = Histogram(
    'noncog_processing_latency_seconds',
    'Processing latency for anomaly detection',
    ['detector_type']
)

class AnomalyDetector:
    """異常検知システム"""
    
    def __init__(self, spark_session: Optional[SparkSession] = None):
        self.spark = spark_session or self._create_spark_session()
        self.logger = structlog.get_logger(__name__)
        
        # 異常検知の閾値設定
        self.thresholds = {
            'zscore': 3.0,  # 標準偏差の倍数
            'iqr_factor': 1.5,  # IQR倍数
            'change_rate': 0.5,  # 変化率の閾値
            'correlation': 0.7  # 相関係数の閾値
        }
    
    def _create_spark_session(self) -> SparkSession:
        """Sparkセッションの作成"""
        return SparkSession.builder \
            .appName("NonCog-AnomalyDetector") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()
    
    def detect_statistical_anomalies(self, df, metrics: List[str]):
        """統計的異常の検出"""
        with PROCESSING_LATENCY.labels('statistical').time():
            # Z-scoreによる異常検出
            zscore_anomalies = df
            for metric in metrics:
                mean_val = df.select(avg(col(metric))).first()[0]
                stddev_val = df.select(stddev(col(metric))).first()[0]
                
                zscore_anomalies = zscore_anomalies.withColumn(
                    f"{metric}_zscore",
                    abs((col(metric) - mean_val) / stddev_val)
                ).withColumn(
                    f"{metric}_is_anomaly",
                    col(f"{metric}_zscore") > self.thresholds['zscore']
                )
            
            # IQRによる異常検出
            for metric in metrics:
                quantiles = df.select(
                    percentile_approx(metric, [0.25, 0.75], 10000)
                ).first()[0]
                q1, q3 = quantiles[0], quantiles[1]
                iqr = q3 - q1
                
                zscore_anomalies = zscore_anomalies.withColumn(
                    f"{metric}_iqr_anomaly",
                    (col(metric) < (q1 - self.thresholds['iqr_factor'] * iqr)) |
                    (col(metric) > (q3 + self.thresholds['iqr_factor'] * iqr))
                )
            
            return zscore_anomalies
    
    def detect_temporal_anomalies(self, df, time_col: str, metrics: List[str]):
        """時系列異常の検出"""
        with PROCESSING_LATENCY.labels('temporal').time():
            # 時間窓の定義
            window_spec = Window \
                .orderBy(time_col) \
                .rangeBetween(-timedelta(hours=1).total_seconds(), 0)
            
            temporal_anomalies = df
            for metric in metrics:
                # 移動平均と標準偏差の計算
                temporal_anomalies = temporal_anomalies \
                    .withColumn(
                        f"{metric}_ma",
                        avg(col(metric)).over(window_spec)
                    ) \
                    .withColumn(
                        f"{metric}_stddev",
                        stddev(col(metric)).over(window_spec)
                    ) \
                    .withColumn(
                        f"{metric}_change_rate",
                        abs((col(metric) - col(f"{metric}_ma")) /
                            greatest(col(f"{metric}_ma"), lit(0.1)))
                    ) \
                    .withColumn(
                        f"{metric}_temporal_anomaly",
                        col(f"{metric}_change_rate") >
                        self.thresholds['change_rate']
                    )
            
            return temporal_anomalies
    
    def detect_correlation_anomalies(self, df, metrics: List[str]):
        """相関異常の検出"""
        with PROCESSING_LATENCY.labels('correlation').time():
            # 特徴量のベクトル化
            assembler = VectorAssembler(
                inputCols=metrics,
                outputCol="features"
            )
            
            vector_df = assembler.transform(df)
            
            # 相関行列の計算
            correlation_matrix = Correlation.corr(vector_df, "features")
            correlation_values = correlation_matrix.first()[0].toArray()
            
            # 相関異常の検出
            correlation_anomalies = df
            for i, metric1 in enumerate(metrics):
                for j, metric2 in enumerate(metrics[i+1:], i+1):
                    if abs(correlation_values[i][j]) > self.thresholds['correlation']:
                        # 期待される相関からの逸脱を検出
                        correlation_anomalies = correlation_anomalies \
                            .withColumn(
                                f"{metric1}_{metric2}_correlation",
                                abs(col(metric1) - col(metric2)) /
                                greatest(col(metric2), lit(0.1))
                            ) \
                            .withColumn(
                                f"{metric1}_{metric2}_correlation_anomaly",
                                col(f"{metric1}_{metric2}_correlation") >
                                self.thresholds['correlation']
                            )
            
            return correlation_anomalies
    
    def detect_pattern_anomalies(self, df, metrics: List[str]):
        """パターン異常の検出"""
        with PROCESSING_LATENCY.labels('pattern').time():
            # 特徴量のベクトル化
            assembler = VectorAssembler(
                inputCols=metrics,
                outputCol="features"
            )
            
            vector_df = assembler.transform(df)
            
            # クラスタリングによるパターン検出
            kmeans = KMeans(k=3, featuresCol="features")
            model = kmeans.fit(vector_df)
            
            # 異常スコアの計算
            pattern_anomalies = model.transform(vector_df) \
                .withColumn(
                    "distance_to_center",
                    self._calculate_distance_udf(
                        "features",
                        array([lit(x) for x in model.clusterCenters()[0].toArray()])
                    )
                ) \
                .withColumn(
                    "pattern_anomaly",
                    col("distance_to_center") >
                    lit(self._calculate_distance_threshold(model))
                )
            
            return pattern_anomalies
    
    def _calculate_distance_udf(self, features_col: str, center_col: str):
        """クラスタ中心からの距離計算UDF"""
        return udf(
            lambda features, center: float(np.sqrt(
                sum((f - c) ** 2 for f, c in zip(features, center))
            )),
            DoubleType()
        )(col(features_col), col(center_col))
    
    def _calculate_distance_threshold(self, model) -> float:
        """距離閾値の計算"""
        distances = []
        for point in model.clusterCenters():
            for other_point in model.clusterCenters():
                if not np.array_equal(point, other_point):
                    distances.append(
                        np.sqrt(sum((p - o) ** 2 for p, o in zip(point, other_point)))
                    )
        return np.mean(distances) * 0.5
    
    def detect_system_anomalies(self, metrics_df):
        """システム全体の異常検出"""
        with PROCESSING_LATENCY.labels('system').time():
            # システムメトリクスの集計
            system_metrics = metrics_df \
                .groupBy("timestamp") \
                .agg(
                    avg("cpu_usage").alias("avg_cpu"),
                    avg("memory_usage").alias("avg_memory"),
                    avg("latency").alias("avg_latency"),
                    count("error_count").alias("total_errors")
                )
            
            # 異常スコアの計算
            system_health = system_metrics \
                .withColumn(
                    "cpu_score",
                    when(col("avg_cpu") > 80, 0.0)
                    .when(col("avg_cpu") > 60, 0.5)
                    .otherwise(1.0)
                ) \
                .withColumn(
                    "memory_score",
                    when(col("avg_memory") > 90, 0.0)
                    .when(col("avg_memory") > 70, 0.5)
                    .otherwise(1.0)
                ) \
                .withColumn(
                    "latency_score",
                    when(col("avg_latency") > 1000, 0.0)
                    .when(col("avg_latency") > 500, 0.5)
                    .otherwise(1.0)
                ) \
                .withColumn(
                    "error_score",
                    when(col("total_errors") > 100, 0.0)
                    .when(col("total_errors") > 50, 0.5)
                    .otherwise(1.0)
                )
            
            # 総合スコアの計算
            system_health = system_health \
                .withColumn(
                    "overall_health",
                    (col("cpu_score") + col("memory_score") +
                     col("latency_score") + col("error_score")) / 4.0
                )
            
            return system_health
    
    def monitor_data_quality(self, df):
        """データ品質のモニタリング"""
        with PROCESSING_LATENCY.labels('data_quality').time():
            # 欠損値の検出
            null_counts = df.select([
                count(when(col(c).isNull(), c)).alias(f"{c}_nulls")
                for c in df.columns
            ])
            
            # 異常値の検出
            numeric_cols = [
                f.name for f in df.schema.fields
                if isinstance(f.dataType, (IntegerType, DoubleType, FloatType))
            ]
            
            bounds = df.select([
                percentile_approx(c, [0.01, 0.99], 10000).alias(f"{c}_bounds")
                for c in numeric_cols
            ]).first()
            
            outlier_counts = df.select([
                count(
                    when(
                        (col(c) < bounds[f"{c}_bounds"][0]) |
                        (col(c) > bounds[f"{c}_bounds"][1]),
                        c
                    )
                ).alias(f"{c}_outliers")
                for c in numeric_cols
            ])
            
            return null_counts.crossJoin(outlier_counts)
    
    def generate_anomaly_report(self, anomalies_df):
        """異常検知レポートの生成"""
        # 異常の集計
        anomaly_summary = anomalies_df \
            .select([
                count(when(col(c).contains("anomaly") & col(c), True)).alias(c)
                for c in anomalies_df.columns
                if "anomaly" in c
            ])
        
        # 重要度によるフィルタリング
        critical_anomalies = []
        warning_anomalies = []
        
        for col_name in [c for c in anomalies_df.columns if "anomaly" in c]:
            count = anomaly_summary.first()[col_name]
            if count > 10:
                critical_anomalies.append({
                    "type": col_name,
                    "count": count
                })
            elif count > 0:
                warning_anomalies.append({
                    "type": col_name,
                    "count": count
                })
        
        # メトリクスの更新
        for anomaly in critical_anomalies:
            ANOMALY_COUNTER.labels(
                type=anomaly["type"],
                severity="critical"
            ).inc(anomaly["count"])
        
        for anomaly in warning_anomalies:
            ANOMALY_COUNTER.labels(
                type=anomaly["type"],
                severity="warning"
            ).inc(anomaly["count"])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "critical_anomalies": critical_anomalies,
            "warning_anomalies": warning_anomalies,
            "total_records": anomalies_df.count(),
            "anomaly_rate": len(critical_anomalies) + len(warning_anomalies)
        }
    
    def process_streaming_anomalies(self, input_stream_df, output_path: str):
        """ストリーミング異常検知の処理"""
        def process_batch(batch_df, batch_id):
            try:
                # メトリクスの抽出
                metrics = [
                    f.name for f in batch_df.schema.fields
                    if isinstance(f.dataType, (IntegerType, DoubleType, FloatType))
                ]
                
                # 各種異常検知の実行
                anomalies = batch_df
                anomalies = self.detect_statistical_anomalies(
                    anomalies, metrics
                )
                anomalies = self.detect_temporal_anomalies(
                    anomalies, "timestamp", metrics
                )
                anomalies = self.detect_correlation_anomalies(
                    anomalies, metrics
                )
                anomalies = self.detect_pattern_anomalies(
                    anomalies, metrics
                )
                
                # システム異常の検出
                system_health = self.detect_system_anomalies(batch_df)
                
                # データ品質のチェック
                data_quality = self.monitor_data_quality(batch_df)
                
                # 異常レポートの生成
                report = self.generate_anomaly_report(anomalies)
                
                # 結果の保存
                anomalies.write \
                    .mode("overwrite") \
                    .parquet(f"{output_path}/anomalies/batch_{batch_id}")
                
                system_health.write \
                    .mode("overwrite") \
                    .parquet(f"{output_path}/system_health/batch_{batch_id}")
                
                data_quality.write \
                    .mode("overwrite") \
                    .parquet(f"{output_path}/data_quality/batch_{batch_id}")
                
                # レポートの保存
                with open(f"{output_path}/reports/batch_{batch_id}.json", "w") as f:
                    json.dump(report, f, indent=2)
                
                self.logger.info(
                    "Anomaly detection completed",
                    batch_id=batch_id,
                    anomalies_found=len(report["critical_anomalies"]) +
                                  len(report["warning_anomalies"])
                )
            
            except Exception as e:
                self.logger.error(
                    "Failed to process anomalies",
                    batch_id=batch_id,
                    error=str(e)
                )
                raise
        
        # ストリーミング処理の開始
        return input_stream_df \
            .writeStream \
            .foreachBatch(process_batch) \
            .outputMode("append") \
            .start()
