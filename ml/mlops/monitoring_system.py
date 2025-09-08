import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prometheus_client
import seaborn as sns
import structlog
import torch
import torch.nn as nn
from prometheus_client import Counter, Gauge, Histogram, Summary
from sklearn.metrics import precision_recall_curve, roc_auc_score

from .model_registry import ModelRegistry, ModelStage


class AlertSeverity(Enum):
    """アラート重要度の定義"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MonitoringConfig:
    """モニタリング設定"""

    model_name: str
    metrics_window: int  # メトリクス収集ウィンドウ（秒）
    alert_thresholds: Dict[str, float]
    performance_thresholds: Dict[str, float]
    resource_thresholds: Dict[str, float]


class MonitoringSystem:
    """モニタリングシステム"""

    def __init__(self, config: MonitoringConfig, model_registry: ModelRegistry):
        self.config = config
        self.model_registry = model_registry

        # メトリクスの初期化
        self._initialize_metrics()

        # アラート履歴
        self.alert_history = []

        # メトリクス履歴
        self.metrics_history = pd.DataFrame()

        # ロガーの設定
        self.logger = structlog.get_logger(__name__)

    def _initialize_metrics(self):
        """メトリクスの初期化"""
        # モデルパフォーマンスメトリクス
        self.prediction_latency = Histogram(
            "model_prediction_latency_seconds",
            "Prediction latency in seconds",
            ["model_name", "version"],
        )

        self.prediction_errors = Counter(
            "model_prediction_errors_total",
            "Total number of prediction errors",
            ["model_name", "version", "error_type"],
        )

        self.prediction_accuracy = Gauge(
            "model_prediction_accuracy",
            "Model prediction accuracy",
            ["model_name", "version"],
        )

        # システムメトリクス
        self.system_memory_usage = Gauge(
            "system_memory_usage_bytes", "System memory usage in bytes", ["component"]
        )

        self.system_cpu_usage = Gauge(
            "system_cpu_usage_percent", "System CPU usage percentage", ["component"]
        )

        self.system_gpu_usage = Gauge(
            "system_gpu_usage_percent", "System GPU usage percentage", ["device"]
        )

        # ビジネスメトリクス
        self.user_satisfaction = Gauge(
            "user_satisfaction_score", "User satisfaction score", ["model_name"]
        )

        self.feature_usage = Counter(
            "feature_usage_total", "Total feature usage count", ["feature_name"]
        )

    async def start_monitoring(self):
        """モニタリングの開始"""
        try:
            # メトリクス収集タスクの開始
            collection_task = asyncio.create_task(self._collect_metrics())

            # アラート監視タスクの開始
            alert_task = asyncio.create_task(self._monitor_alerts())

            # レポート生成タスクの開始
            report_task = asyncio.create_task(self._generate_reports())

            # 全タスクの実行
            await asyncio.gather(collection_task, alert_task, report_task)

        except Exception as e:
            self.logger.error("Monitoring system failed", error=str(e))
            raise

    async def _collect_metrics(self):
        """メトリクスの収集"""
        while True:
            try:
                # モデルメトリクスの収集
                model_metrics = await self._collect_model_metrics()

                # システムメトリクスの収集
                system_metrics = await self._collect_system_metrics()

                # ビジネスメトリクスの収集
                business_metrics = await self._collect_business_metrics()

                # メトリクスの結合
                metrics = {
                    **model_metrics,
                    **system_metrics,
                    **business_metrics,
                    "timestamp": datetime.now(),
                }

                # メトリクス履歴の更新
                self.metrics_history = pd.concat(
                    [self.metrics_history, pd.DataFrame([metrics])]
                )

                # 古いメトリクスの削除
                cutoff_time = datetime.now() - pd.Timedelta(
                    seconds=self.config.metrics_window
                )
                self.metrics_history = self.metrics_history[
                    self.metrics_history["timestamp"] > cutoff_time
                ]

                # スリープ
                await asyncio.sleep(10)  # 10秒ごとに収集

            except Exception as e:
                self.logger.error("Failed to collect metrics", error=str(e))
                await asyncio.sleep(60)  # エラー時は1分待機

    async def _collect_model_metrics(self) -> Dict[str, float]:
        """モデルメトリクスの収集"""
        metrics = {}

        try:
            # 現在のモデルバージョンの取得
            model_info = self.model_registry.get_model_history(self.config.model_name)[
                -1
            ]

            # 予測レイテンシーの計測
            with self.prediction_latency.labels(
                self.config.model_name, model_info["version"]
            ).time():
                # テスト予測の実行
                await self._test_prediction()

            # 精度メトリクスの計算
            accuracy = await self._calculate_accuracy()
            self.prediction_accuracy.labels(
                self.config.model_name, model_info["version"]
            ).set(accuracy)

            metrics.update(
                {
                    "prediction_latency": self.prediction_latency._sum.get(),
                    "prediction_accuracy": accuracy,
                }
            )

        except Exception as e:
            self.logger.error("Failed to collect model metrics", error=str(e))
            self.prediction_errors.labels(
                self.config.model_name, model_info["version"], "collection_error"
            ).inc()

        return metrics

    async def _collect_system_metrics(self) -> Dict[str, float]:
        """システムメトリクスの収集"""
        metrics = {}

        try:
            # メモリ使用率の収集
            memory_info = await self._get_memory_info()
            for component, usage in memory_info.items():
                self.system_memory_usage.labels(component).set(usage)
                metrics[f"memory_usage_{component}"] = usage

            # CPU使用率の収集
            cpu_info = await self._get_cpu_info()
            for component, usage in cpu_info.items():
                self.system_cpu_usage.labels(component).set(usage)
                metrics[f"cpu_usage_{component}"] = usage

            # GPU使用率の収集
            if torch.cuda.is_available():
                gpu_info = await self._get_gpu_info()
                for device, usage in gpu_info.items():
                    self.system_gpu_usage.labels(device).set(usage)
                    metrics[f"gpu_usage_{device}"] = usage

        except Exception as e:
            self.logger.error("Failed to collect system metrics", error=str(e))

        return metrics

    async def _collect_business_metrics(self) -> Dict[str, float]:
        """ビジネスメトリクスの収集"""
        metrics = {}

        try:
            # ユーザー満足度の収集
            satisfaction = await self._get_user_satisfaction()
            self.user_satisfaction.labels(self.config.model_name).set(satisfaction)
            metrics["user_satisfaction"] = satisfaction

            # 機能使用状況の収集
            feature_usage = await self._get_feature_usage()
            for feature, count in feature_usage.items():
                self.feature_usage.labels(feature).inc(count)
                metrics[f"feature_usage_{feature}"] = count

        except Exception as e:
            self.logger.error("Failed to collect business metrics", error=str(e))

        return metrics

    async def _monitor_alerts(self):
        """アラートの監視"""
        while True:
            try:
                # メトリクスの分析
                alerts = await self._analyze_metrics()

                # アラートの発行
                for alert in alerts:
                    await self._issue_alert(alert)

                # スリープ
                await asyncio.sleep(60)  # 1分ごとに監視

            except Exception as e:
                self.logger.error("Failed to monitor alerts", error=str(e))
                await asyncio.sleep(300)  # エラー時は5分待機

    async def _analyze_metrics(self) -> List[Dict[str, Any]]:
        """メトリクスの分析"""
        alerts = []

        try:
            # パフォーマンスアラート
            if len(self.metrics_history) > 0:
                recent_metrics = self.metrics_history.iloc[-1]

                # 精度の低下
                if (
                    recent_metrics["prediction_accuracy"]
                    < self.config.performance_thresholds["min_accuracy"]
                ):
                    alerts.append(
                        {
                            "type": "accuracy_degradation",
                            "severity": AlertSeverity.ERROR,
                            "message": "Model accuracy has dropped below threshold",
                            "value": recent_metrics["prediction_accuracy"],
                        }
                    )

                # レイテンシーの上昇
                if (
                    recent_metrics["prediction_latency"]
                    > self.config.performance_thresholds["max_latency"]
                ):
                    alerts.append(
                        {
                            "type": "high_latency",
                            "severity": AlertSeverity.WARNING,
                            "message": "Prediction latency is above threshold",
                            "value": recent_metrics["prediction_latency"],
                        }
                    )

            # リソースアラート
            for component in ["model", "api", "database"]:
                memory_key = f"memory_usage_{component}"
                cpu_key = f"cpu_usage_{component}"

                if (
                    memory_key in recent_metrics
                    and recent_metrics[memory_key]
                    > self.config.resource_thresholds["max_memory"]
                ):
                    alerts.append(
                        {
                            "type": "high_memory_usage",
                            "severity": AlertSeverity.WARNING,
                            "message": f"High memory usage in {component}",
                            "value": recent_metrics[memory_key],
                        }
                    )

                if (
                    cpu_key in recent_metrics
                    and recent_metrics[cpu_key]
                    > self.config.resource_thresholds["max_cpu"]
                ):
                    alerts.append(
                        {
                            "type": "high_cpu_usage",
                            "severity": AlertSeverity.WARNING,
                            "message": f"High CPU usage in {component}",
                            "value": recent_metrics[cpu_key],
                        }
                    )

            # ビジネスアラート
            if (
                recent_metrics["user_satisfaction"]
                < self.config.alert_thresholds["min_satisfaction"]
            ):
                alerts.append(
                    {
                        "type": "low_satisfaction",
                        "severity": AlertSeverity.ERROR,
                        "message": "User satisfaction is below threshold",
                        "value": recent_metrics["user_satisfaction"],
                    }
                )

        except Exception as e:
            self.logger.error("Failed to analyze metrics", error=str(e))

        return alerts

    async def _issue_alert(self, alert: Dict[str, Any]):
        """アラートの発行"""
        try:
            # アラート履歴への追加
            alert["timestamp"] = datetime.now()
            self.alert_history.append(alert)

            # ログへの記録
            self.logger.warning(
                "Alert issued",
                type=alert["type"],
                severity=alert["severity"].value,
                message=alert["message"],
                value=alert["value"],
            )

            # 重要度に応じた通知
            if alert["severity"] in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                await self._send_notification(alert)

        except Exception as e:
            self.logger.error("Failed to issue alert", alert=alert, error=str(e))

    async def _generate_reports(self):
        """レポートの生成"""
        while True:
            try:
                # 日次レポートの生成
                if self._should_generate_daily_report():
                    report = await self._generate_daily_report()
                    await self._save_report(report, "daily")

                # 週次レポートの生成
                if self._should_generate_weekly_report():
                    report = await self._generate_weekly_report()
                    await self._save_report(report, "weekly")

                # スリープ
                await asyncio.sleep(3600)  # 1時間ごとにチェック

            except Exception as e:
                self.logger.error("Failed to generate reports", error=str(e))
                await asyncio.sleep(3600)

    async def _generate_daily_report(self) -> Dict[str, Any]:
        """日次レポートの生成"""
        report = {
            "timestamp": datetime.now(),
            "period": "daily",
            "model_name": self.config.model_name,
            "metrics_summary": {},
            "alerts_summary": {},
            "visualizations": {},
        }

        # メトリクスサマリーの計算
        daily_metrics = self.metrics_history[
            self.metrics_history["timestamp"] > (datetime.now() - pd.Timedelta(days=1))
        ]

        report["metrics_summary"] = {
            "accuracy": {
                "mean": daily_metrics["prediction_accuracy"].mean(),
                "std": daily_metrics["prediction_accuracy"].std(),
                "min": daily_metrics["prediction_accuracy"].min(),
                "max": daily_metrics["prediction_accuracy"].max(),
            },
            "latency": {
                "mean": daily_metrics["prediction_latency"].mean(),
                "p95": daily_metrics["prediction_latency"].quantile(0.95),
                "p99": daily_metrics["prediction_latency"].quantile(0.99),
            },
        }

        # アラートサマリーの計算
        daily_alerts = [
            alert
            for alert in self.alert_history
            if alert["timestamp"] > datetime.now() - pd.Timedelta(days=1)
        ]

        report["alerts_summary"] = {
            "total_count": len(daily_alerts),
            "by_severity": {
                severity.value: len(
                    [a for a in daily_alerts if a["severity"] == severity]
                )
                for severity in AlertSeverity
            },
            "by_type": {},
        }

        # 可視化の生成
        report["visualizations"] = await self._generate_visualizations(daily_metrics)

        return report

    async def _generate_weekly_report(self) -> Dict[str, Any]:
        """週次レポートの生成"""
        report = {
            "timestamp": datetime.now(),
            "period": "weekly",
            "model_name": self.config.model_name,
            "metrics_summary": {},
            "alerts_summary": {},
            "visualizations": {},
            "trends": {},
        }

        # メトリクスサマリーの計算
        weekly_metrics = self.metrics_history[
            self.metrics_history["timestamp"] > (datetime.now() - pd.Timedelta(weeks=1))
        ]

        # 日ごとの集計
        daily_stats = weekly_metrics.groupby(weekly_metrics["timestamp"].dt.date).agg(
            {
                "prediction_accuracy": ["mean", "std"],
                "prediction_latency": ["mean", "p95"],
                "user_satisfaction": "mean",
            }
        )

        report["metrics_summary"] = {
            "by_day": daily_stats.to_dict(),
            "overall": {
                "accuracy_trend": self._calculate_trend(
                    daily_stats["prediction_accuracy"]["mean"]
                ),
                "latency_trend": self._calculate_trend(
                    daily_stats["prediction_latency"]["mean"]
                ),
            },
        }

        # アラート分析
        weekly_alerts = [
            alert
            for alert in self.alert_history
            if alert["timestamp"] > datetime.now() - pd.Timedelta(weeks=1)
        ]

        report["alerts_summary"] = {
            "total_count": len(weekly_alerts),
            "by_day": {},
            "by_severity": {
                severity.value: len(
                    [a for a in weekly_alerts if a["severity"] == severity]
                )
                for severity in AlertSeverity
            },
            "trend": self._analyze_alert_trend(weekly_alerts),
        }

        # トレンド分析
        report["trends"] = {
            "performance_trend": await self._analyze_performance_trend(weekly_metrics),
            "resource_trend": await self._analyze_resource_trend(weekly_metrics),
            "business_trend": await self._analyze_business_trend(weekly_metrics),
        }

        return report

    def _calculate_trend(self, series: pd.Series) -> Dict[str, float]:
        """トレンドの計算"""
        if len(series) < 2:
            return {"slope": 0.0, "change_rate": 0.0}

        # 線形回帰による傾きの計算
        x = np.arange(len(series))
        slope, _ = np.polyfit(x, series, 1)

        # 変化率の計算
        change_rate = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]

        return {"slope": float(slope), "change_rate": float(change_rate)}

    async def _analyze_performance_trend(self, metrics: pd.DataFrame) -> Dict[str, Any]:
        """パフォーマンストレンドの分析"""
        return {
            "accuracy": self._calculate_trend(metrics["prediction_accuracy"]),
            "latency": self._calculate_trend(metrics["prediction_latency"]),
        }

    async def _analyze_resource_trend(self, metrics: pd.DataFrame) -> Dict[str, Any]:
        """リソーストレンドの分析"""
        resource_trends = {}

        for component in ["model", "api", "database"]:
            memory_key = f"memory_usage_{component}"
            cpu_key = f"cpu_usage_{component}"

            if memory_key in metrics.columns:
                resource_trends[f"{component}_memory"] = self._calculate_trend(
                    metrics[memory_key]
                )

            if cpu_key in metrics.columns:
                resource_trends[f"{component}_cpu"] = self._calculate_trend(
                    metrics[cpu_key]
                )

        return resource_trends

    async def _analyze_business_trend(self, metrics: pd.DataFrame) -> Dict[str, Any]:
        """ビジネストレンドの分析"""
        return {
            "satisfaction": self._calculate_trend(metrics["user_satisfaction"]),
            "feature_usage": {
                feature: self._calculate_trend(metrics[f"feature_usage_{feature}"])
                for feature in ["learning", "feedback", "collaboration"]
                if f"feature_usage_{feature}" in metrics.columns
            },
        }

    def _analyze_alert_trend(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """アラートトレンドの分析"""
        if not alerts:
            return {}

        # 日付ごとのアラート数
        alert_counts = (
            pd.Series([alert["timestamp"].date() for alert in alerts])
            .value_counts()
            .sort_index()
        )

        return {
            "daily_counts": alert_counts.to_dict(),
            "trend": self._calculate_trend(alert_counts),
        }

    async def _generate_visualizations(self, metrics: pd.DataFrame) -> Dict[str, str]:
        """可視化の生成"""
        visualizations = {}

        try:
            # 精度の時系列プロット
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=metrics, x="timestamp", y="prediction_accuracy", label="Accuracy"
            )
            plt.title("Model Accuracy Over Time")
            plt.xticks(rotation=45)
            visualizations["accuracy_plot"] = self._save_plot()

            # レイテンシーの分布プロット
            plt.figure(figsize=(10, 6))
            sns.histplot(data=metrics, x="prediction_latency", bins=30)
            plt.title("Prediction Latency Distribution")
            visualizations["latency_plot"] = self._save_plot()

            # リソース使用率のヒートマップ
            resource_cols = [col for col in metrics.columns if "usage" in col]
            if resource_cols:
                plt.figure(figsize=(12, 8))
                sns.heatmap(metrics[resource_cols].corr(), annot=True, cmap="coolwarm")
                plt.title("Resource Usage Correlation")
                visualizations["resource_plot"] = self._save_plot()

        except Exception as e:
            self.logger.error("Failed to generate visualizations", error=str(e))

        return visualizations

    def _save_plot(self) -> str:
        """プロットの保存"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()

        return base64.b64encode(buffer.getvalue()).decode()

    async def _save_report(self, report: Dict[str, Any], report_type: str):
        """レポートの保存"""
        try:
            # レポートディレクトリの作成
            report_dir = Path(f"reports/{self.config.model_name}/{report_type}")
            report_dir.mkdir(parents=True, exist_ok=True)

            # レポートの保存
            timestamp = report["timestamp"].strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"report_{timestamp}.json"

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info(
                "Report saved", report_type=report_type, path=str(report_path)
            )

        except Exception as e:
            self.logger.error(
                "Failed to save report", report_type=report_type, error=str(e)
            )

    def _should_generate_daily_report(self) -> bool:
        """日次レポート生成の判定"""
        if not hasattr(self, "_last_daily_report"):
            self._last_daily_report = datetime.min

        now = datetime.now()
        should_generate = (
            now.date() > self._last_daily_report.date() and now.hour >= 0  # 0時以降
        )

        if should_generate:
            self._last_daily_report = now

        return should_generate

    def _should_generate_weekly_report(self) -> bool:
        """週次レポート生成の判定"""
        if not hasattr(self, "_last_weekly_report"):
            self._last_weekly_report = datetime.min

        now = datetime.now()
        should_generate = (
            now.isocalendar()[1] > self._last_weekly_report.isocalendar()[1]
            and now.weekday() == 0  # 月曜日
            and now.hour >= 0  # 0時以降
        )

        if should_generate:
            self._last_weekly_report = now

        return should_generate

    async def _test_prediction(self):
        """テスト予測の実行"""
        # テスト予測のロジックをここに実装
        await asyncio.sleep(0.1)

    async def _calculate_accuracy(self) -> float:
        """精度の計算"""
        # 精度計算のロジックをここに実装
        return 0.95

    async def _get_memory_info(self) -> Dict[str, float]:
        """メモリ情報の取得"""
        # メモリ情報取得のロジックをここに実装
        return {
            "model": 500 * 1024 * 1024,  # 500MB
            "api": 200 * 1024 * 1024,  # 200MB
            "database": 1024 * 1024 * 1024,  # 1GB
        }

    async def _get_cpu_info(self) -> Dict[str, float]:
        """CPU情報の取得"""
        # CPU情報取得のロジックをここに実装
        return {"model": 30.0, "api": 20.0, "database": 40.0}  # 30%  # 20%  # 40%

    async def _get_gpu_info(self) -> Dict[str, float]:
        """GPU情報の取得"""
        # GPU情報取得のロジックをここに実装
        return {"gpu0": 50.0, "gpu1": 40.0}  # 50%  # 40%

    async def _get_user_satisfaction(self) -> float:
        """ユーザー満足度の取得"""
        # 満足度取得のロジックをここに実装
        return 4.5  # 5段階評価

    async def _get_feature_usage(self) -> Dict[str, int]:
        """機能使用状況の取得"""
        # 使用状況取得のロジックをここに実装
        return {"learning": 1000, "feedback": 500, "collaboration": 300}

    async def _send_notification(self, alert: Dict[str, Any]):
        """通知の送信"""
        # 通知送信のロジックをここに実装
        pass
