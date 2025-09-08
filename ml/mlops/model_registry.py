import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import git
import mlflow
import mlflow.pytorch
import numpy as np
import structlog
import torch
import torch.nn as nn
import yaml
from mlflow.tracking import MlflowClient


class ModelStage(Enum):
    """モデルステージの定義"""

    DEVELOPMENT = "Development"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass
class ModelMetadata:
    """モデルメタデータ"""

    name: str
    version: str
    stage: ModelStage
    framework: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: Dict[str, str]


class ModelRegistry:
    """モデル管理システム"""

    def __init__(self, tracking_uri: str, registry_uri: str):
        # MLflowの設定
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)

        self.client = MlflowClient()

        # Gitリポジトリの設定
        self.repo = git.Repo(search_parent_directories=True)

        # ロガーの設定
        self.logger = structlog.get_logger(__name__)

    def register_model(self, model: nn.Module, metadata: ModelMetadata) -> str:
        """モデルの登録"""
        try:
            # MLflowの実験を開始
            with mlflow.start_run() as run:
                # Gitコミット情報の記録
                commit = self.repo.head.commit
                mlflow.set_tags(
                    {
                        "git_commit": commit.hexsha,
                        "git_branch": self.repo.active_branch.name,
                        "git_repo": self.repo.remotes.origin.url,
                    }
                )

                # メタデータの記録
                mlflow.log_params(metadata.parameters)
                mlflow.log_metrics(metadata.metrics)
                mlflow.set_tags(metadata.tags)

                # スキーマ情報の保存
                schema_path = Path("artifacts/schema.yaml")
                schema_path.parent.mkdir(parents=True, exist_ok=True)

                schema = {
                    "input_schema": metadata.input_schema,
                    "output_schema": metadata.output_schema,
                }

                with open(schema_path, "w") as f:
                    yaml.dump(schema, f)

                mlflow.log_artifact(schema_path)

                # モデルの保存
                mlflow.pytorch.log_model(
                    model, "model", registered_model_name=metadata.name
                )

                # バージョンの作成
                version = self.client.create_model_version(
                    name=metadata.name,
                    source=f"runs:/{run.info.run_id}/model",
                    run_id=run.info.run_id,
                )

                # ステージの設定
                self.client.transition_model_version_stage(
                    name=metadata.name,
                    version=version.version,
                    stage=metadata.stage.value,
                )

                self.logger.info(
                    "Model registered successfully",
                    model_name=metadata.name,
                    version=version.version,
                    stage=metadata.stage.value,
                )

                return version.version

        except Exception as e:
            self.logger.error(
                "Failed to register model", model_name=metadata.name, error=str(e)
            )
            raise

    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None,
    ) -> Tuple[nn.Module, ModelMetadata]:
        """モデルの読み込み"""
        try:
            # バージョンまたはステージの指定を確認
            if version is None and stage is None:
                stage = ModelStage.PRODUCTION

            if stage is not None:
                model_uri = f"models:/{name}/{stage.value}"
            else:
                model_uri = f"models:/{name}/{version}"

            # モデルの読み込み
            model = mlflow.pytorch.load_model(model_uri)

            # メタデータの取得
            if stage is not None:
                version_info = self.client.get_latest_versions(
                    name, stages=[stage.value]
                )[0]
            else:
                version_info = self.client.get_model_version(name, version)

            run = self.client.get_run(version_info.run_id)

            # スキーマ情報の読み込み
            artifact_path = self.client.download_artifacts(
                run.info.run_id, "artifacts/schema.yaml"
            )

            with open(artifact_path) as f:
                schema = yaml.safe_load(f)

            metadata = ModelMetadata(
                name=name,
                version=version_info.version,
                stage=ModelStage(version_info.current_stage),
                framework="pytorch",
                input_schema=schema["input_schema"],
                output_schema=schema["output_schema"],
                metrics=run.data.metrics,
                parameters=run.data.params,
                tags=run.data.tags,
            )

            return model, metadata

        except Exception as e:
            self.logger.error(
                "Failed to load model",
                model_name=name,
                version=version,
                stage=stage,
                error=str(e),
            )
            raise

    def transition_model_stage(
        self, name: str, version: str, stage: ModelStage
    ) -> None:
        """モデルステージの遷移"""
        try:
            # 現在のステージを確認
            current_version = self.client.get_model_version(name, version)
            current_stage = ModelStage(current_version.current_stage)

            # ステージ遷移のバリデーション
            if not self._validate_stage_transition(current_stage, stage):
                raise ValueError(
                    f"Invalid stage transition: {current_stage.value} -> {stage.value}"
                )

            # ステージの更新
            self.client.transition_model_version_stage(
                name=name, version=version, stage=stage.value
            )

            self.logger.info(
                "Model stage transitioned",
                model_name=name,
                version=version,
                from_stage=current_stage.value,
                to_stage=stage.value,
            )

        except Exception as e:
            self.logger.error(
                "Failed to transition model stage",
                model_name=name,
                version=version,
                stage=stage,
                error=str(e),
            )
            raise

    def _validate_stage_transition(
        self, current_stage: ModelStage, new_stage: ModelStage
    ) -> bool:
        """ステージ遷移の検証"""
        # 遷移ルールの定義
        valid_transitions = {
            ModelStage.DEVELOPMENT: {ModelStage.STAGING, ModelStage.ARCHIVED},
            ModelStage.STAGING: {
                ModelStage.PRODUCTION,
                ModelStage.DEVELOPMENT,
                ModelStage.ARCHIVED,
            },
            ModelStage.PRODUCTION: {ModelStage.STAGING, ModelStage.ARCHIVED},
            ModelStage.ARCHIVED: {ModelStage.DEVELOPMENT},
        }

        return new_stage in valid_transitions[current_stage]

    def compare_models(self, name: str, versions: List[str]) -> Dict[str, Any]:
        """モデルの比較"""
        try:
            comparison = {"metrics": {}, "parameters": {}, "tags": {}}

            # 各バージョンの情報を収集
            for version in versions:
                version_info = self.client.get_model_version(name, version)
                run = self.client.get_run(version_info.run_id)

                comparison["metrics"][version] = run.data.metrics
                comparison["parameters"][version] = run.data.params
                comparison["tags"][version] = run.data.tags

            return comparison

        except Exception as e:
            self.logger.error(
                "Failed to compare models",
                model_name=name,
                versions=versions,
                error=str(e),
            )
            raise

    def delete_model_version(self, name: str, version: str) -> None:
        """モデルバージョンの削除"""
        try:
            # バージョンの存在確認
            version_info = self.client.get_model_version(name, version)

            # Productionステージのモデルは削除不可
            if version_info.current_stage == ModelStage.PRODUCTION.value:
                raise ValueError("Cannot delete model in Production stage")

            # バージョンの削除
            self.client.delete_model_version(name=name, version=version)

            self.logger.info("Model version deleted", model_name=name, version=version)

        except Exception as e:
            self.logger.error(
                "Failed to delete model version",
                model_name=name,
                version=version,
                error=str(e),
            )
            raise

    def get_model_history(self, name: str) -> List[Dict[str, Any]]:
        """モデルの履歴取得"""
        try:
            history = []

            # 全バージョンの取得
            versions = self.client.search_model_versions(f"name='{name}'")

            for version in versions:
                run = self.client.get_run(version.run_id)

                history.append(
                    {
                        "version": version.version,
                        "stage": version.current_stage,
                        "creation_timestamp": version.creation_timestamp,
                        "last_updated_timestamp": version.last_updated_timestamp,
                        "metrics": run.data.metrics,
                        "parameters": run.data.params,
                        "tags": run.data.tags,
                        "git_commit": run.data.tags.get("git_commit"),
                        "git_branch": run.data.tags.get("git_branch"),
                    }
                )

            return history

        except Exception as e:
            self.logger.error(
                "Failed to get model history", model_name=name, error=str(e)
            )
            raise

    def export_model(self, name: str, version: str, export_path: str) -> None:
        """モデルのエクスポート"""
        try:
            # モデルとメタデータの読み込み
            model, metadata = self.load_model(name, version)

            # エクスポートディレクトリの作成
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)

            # モデルの保存
            torch.save(model.state_dict(), export_dir / "model.pth")

            # メタデータの保存
            with open(export_dir / "metadata.yaml", "w") as f:
                yaml.dump(metadata.__dict__, f)

            self.logger.info(
                "Model exported successfully",
                model_name=name,
                version=version,
                export_path=export_path,
            )

        except Exception as e:
            self.logger.error(
                "Failed to export model",
                model_name=name,
                version=version,
                export_path=export_path,
                error=str(e),
            )
            raise

    def import_model(self, import_path: str) -> Tuple[str, str]:
        """モデルのインポート"""
        try:
            import_dir = Path(import_path)

            # メタデータの読み込み
            with open(import_dir / "metadata.yaml") as f:
                metadata_dict = yaml.safe_load(f)
                metadata = ModelMetadata(**metadata_dict)

            # モデルの読み込み
            model = self._create_model_instance(metadata)
            model.load_state_dict(torch.load(import_dir / "model.pth"))

            # モデルの登録
            version = self.register_model(model, metadata)

            self.logger.info(
                "Model imported successfully",
                model_name=metadata.name,
                version=version,
                import_path=import_path,
            )

            return metadata.name, version

        except Exception as e:
            self.logger.error(
                "Failed to import model", import_path=import_path, error=str(e)
            )
            raise

    def _create_model_instance(self, metadata: ModelMetadata) -> nn.Module:
        """モデルインスタンスの作成"""
        # メタデータに基づいてモデルアーキテクチャを再構築
        if metadata.framework != "pytorch":
            raise ValueError(f"Unsupported framework: {metadata.framework}")

        # モデルの構築ロジックをここに実装
        # 例: DQNモデルの場合
        if "dqn" in metadata.name.lower():
            from ..models.advanced_dqn import DuelingDQN

            return DuelingDQN(
                state_dim=metadata.parameters.get("state_dim", 8),
                action_dim=metadata.parameters.get("action_dim", 60),
                hidden_dim=metadata.parameters.get("hidden_dim", 256),
            )

        raise ValueError(f"Unknown model type: {metadata.name}")
