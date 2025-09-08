import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import optuna
import structlog
import torch
import torch.nn as nn
import yaml
from optuna.trial import Trial

from .model_registry import ModelMetadata, ModelRegistry, ModelStage


class DatasetType(Enum):
    """データセットタイプの定義"""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class TrainingConfig:
    """学習設定"""

    experiment_name: str
    model_name: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    optimizer_type: str
    loss_function: str
    early_stopping_patience: int
    early_stopping_threshold: float
    num_trials: int  # ハイパーパラメータ最適化の試行回数


class TrainingPipeline:
    """自動学習パイプライン"""

    def __init__(self, config: TrainingConfig, model_registry: ModelRegistry):
        self.config = config
        self.model_registry = model_registry

        # MLflowの実験設定
        self.experiment = mlflow.get_experiment_by_name(config.experiment_name)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(config.experiment_name)
        else:
            self.experiment_id = self.experiment.experiment_id

        # ロガーの設定
        self.logger = structlog.get_logger(__name__)

    def run_pipeline(
        self, train_data: Any, val_data: Any, test_data: Optional[Any] = None
    ) -> str:
        """パイプラインの実行"""
        try:
            # データの検証
            self._validate_data(train_data, DatasetType.TRAIN)
            self._validate_data(val_data, DatasetType.VALIDATION)
            if test_data is not None:
                self._validate_data(test_data, DatasetType.TEST)

            # ハイパーパラメータの最適化
            best_params = self._optimize_hyperparameters(train_data, val_data)

            # 最適なモデルの学習
            model = self._train_model(train_data, val_data, best_params)

            # モデルの評価
            metrics = self._evaluate_model(model, test_data or val_data)

            # モデルの登録
            version = self._register_model(model, best_params, metrics)

            self.logger.info(
                "Training pipeline completed successfully",
                model_name=self.config.model_name,
                version=version,
            )

            return version

        except Exception as e:
            self.logger.error("Training pipeline failed", error=str(e))
            raise

    def _validate_data(self, data: Any, dataset_type: DatasetType):
        """データの検証"""
        # データセットの基本チェック
        if data is None:
            raise ValueError(f"{dataset_type.value} dataset is None")

        # データ型の確認
        if not isinstance(
            data, (torch.utils.data.Dataset, torch.utils.data.DataLoader)
        ):
            raise TypeError(f"Unsupported dataset type: {type(data)}")

        # データサイズの確認
        if len(data) == 0:
            raise ValueError(f"{dataset_type.value} dataset is empty")

    def _optimize_hyperparameters(
        self, train_data: Any, val_data: Any
    ) -> Dict[str, Any]:
        """ハイパーパラメータの最適化"""

        def objective(trial: Trial) -> float:
            # ハイパーパラメータの定義
            params = {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [16, 32, 64, 128]
                ),
                "hidden_dim": trial.suggest_categorical(
                    "hidden_dim", [64, 128, 256, 512]
                ),
                "dropout_rate": trial.suggest_uniform("dropout_rate", 0.1, 0.5),
            }

            # モデルの学習と評価
            model = self._train_model(train_data, val_data, params)
            metrics = self._evaluate_model(model, val_data)

            return metrics["validation_loss"]

        # Optunaによる最適化
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config.num_trials)

        return study.best_params

    def _train_model(
        self, train_data: Any, val_data: Any, params: Dict[str, Any]
    ) -> nn.Module:
        """モデルの学習"""
        # MLflow実験の開始
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            # パラメータのログ
            mlflow.log_params(params)

            # モデルの構築
            model = self._create_model(params)
            model.to(self.device)

            # オプティマイザとロス関数の設定
            optimizer = self._create_optimizer(model, params)
            criterion = self._create_criterion()

            # 学習ループ
            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(self.config.num_epochs):
                # 学習フェーズ
                train_loss = self._train_epoch(model, train_data, optimizer, criterion)

                # 検証フェーズ
                val_loss = self._validate_epoch(model, val_data, criterion)

                # メトリクスのログ
                mlflow.log_metrics(
                    {"train_loss": train_loss, "validation_loss": val_loss}, step=epoch
                )

                # 早期停止の確認
                if val_loss < best_val_loss - self.config.early_stopping_threshold:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # ベストモデルの保存
                    torch.save(model.state_dict(), "best_model.pth")
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(
                        "Early stopping triggered",
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                    )
                    break

            # ベストモデルの読み込み
            model.load_state_dict(torch.load("best_model.pth"))

            return model

    def _train_epoch(
        self,
        model: nn.Module,
        train_data: Any,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """1エポックの学習"""
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_data:
            optimizer.zero_grad()

            # バッチデータの処理
            inputs, targets = self._prepare_batch(batch)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 逆伝播と最適化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(
        self, model: nn.Module, val_data: Any, criterion: nn.Module
    ) -> float:
        """1エポックの検証"""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_data:
                # バッチデータの処理
                inputs, targets = self._prepare_batch(batch)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _evaluate_model(self, model: nn.Module, test_data: Any) -> Dict[str, float]:
        """モデルの評価"""
        model.eval()
        metrics = {"test_loss": 0.0, "accuracy": 0.0}

        criterion = self._create_criterion()
        num_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_data:
                # バッチデータの処理
                inputs, targets = self._prepare_batch(batch)
                outputs = model(inputs)

                # 損失の計算
                loss = criterion(outputs, targets)
                metrics["test_loss"] += loss.item()

                # 精度の計算
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                num_batches += 1

        metrics["test_loss"] /= num_batches
        metrics["accuracy"] = correct / total

        return metrics

    def _create_model(self, params: Dict[str, Any]) -> nn.Module:
        """モデルの作成"""
        # モデルの構築ロジックをここに実装
        # 例: DQNモデルの場合
        if "dqn" in self.config.model_name.lower():
            from ..models.advanced_dqn import DuelingDQN

            return DuelingDQN(
                state_dim=8, action_dim=60, hidden_dim=params["hidden_dim"]
            )

        raise ValueError(f"Unknown model type: {self.config.model_name}")

    def _create_optimizer(
        self, model: nn.Module, params: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        """オプティマイザの作成"""
        if self.config.optimizer_type.lower() == "adam":
            return torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        elif self.config.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                model.parameters(), lr=params["learning_rate"], momentum=0.9
            )

        raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

    def _create_criterion(self) -> nn.Module:
        """損失関数の作成"""
        if self.config.loss_function.lower() == "mse":
            return nn.MSELoss()
        elif self.config.loss_function.lower() == "crossentropy":
            return nn.CrossEntropyLoss()

        raise ValueError(f"Unknown loss function: {self.config.loss_function}")

    def _prepare_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """バッチデータの準備"""
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, targets = batch
            return inputs.to(self.device), targets.to(self.device)

        raise ValueError(f"Unsupported batch format: {type(batch)}")

    def _register_model(
        self, model: nn.Module, params: Dict[str, Any], metrics: Dict[str, float]
    ) -> str:
        """モデルの登録"""
        # メタデータの作成
        metadata = ModelMetadata(
            name=self.config.model_name,
            version="",  # バージョンは登録時に自動生成
            stage=ModelStage.DEVELOPMENT,
            framework="pytorch",
            input_schema=self._get_input_schema(),
            output_schema=self._get_output_schema(),
            metrics=metrics,
            parameters=params,
            tags={"experiment_id": self.experiment_id, "pipeline_version": "1.0.0"},
        )

        # モデルの登録
        version = self.model_registry.register_model(model, metadata)

        return version

    def _get_input_schema(self) -> Dict[str, Any]:
        """入力スキーマの取得"""
        # モデルタイプに応じたスキーマを定義
        if "dqn" in self.config.model_name.lower():
            return {"state": {"type": "tensor", "shape": [None, 8], "dtype": "float32"}}

        raise ValueError(f"Unknown model type: {self.config.model_name}")

    def _get_output_schema(self) -> Dict[str, Any]:
        """出力スキーマの取得"""
        # モデルタイプに応じたスキーマを定義
        if "dqn" in self.config.model_name.lower():
            return {
                "q_values": {"type": "tensor", "shape": [None, 60], "dtype": "float32"}
            }

        raise ValueError(f"Unknown model type: {self.config.model_name}")
