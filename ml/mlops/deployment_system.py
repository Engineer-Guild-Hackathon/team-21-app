import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging
import json
from datetime import datetime
import structlog
from pathlib import Path
import docker
import kubernetes
from kubernetes import client, config
import yaml
from enum import Enum
from .model_registry import ModelRegistry, ModelStage

class DeploymentStrategy(Enum):
    """デプロイメント戦略の定義"""
    ROLLING = "rolling"
    BLUE_GREEN = "blue-green"
    CANARY = "canary"

@dataclass
class DeploymentConfig:
    """デプロイメント設定"""
    model_name: str
    version: str
    strategy: DeploymentStrategy
    resource_requirements: Dict[str, str]
    scaling_config: Dict[str, Any]
    health_check_config: Dict[str, Any]
    rollback_config: Dict[str, Any]

class DeploymentSystem:
    """デプロイメントシステム"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        
        # Kubernetesクライアントの設定
        config.load_kube_config()
        self.k8s_client = client.ApiClient()
        self.k8s_apps = client.AppsV1Api()
        self.k8s_core = client.CoreV1Api()
        
        # Dockerクライアントの設定
        self.docker_client = docker.from_env()
        
        # ロガーの設定
        self.logger = structlog.get_logger(__name__)
    
    def deploy_model(self, deployment_config: DeploymentConfig) -> bool:
        """モデルのデプロイ"""
        try:
            # モデルの読み込み
            model, metadata = self.model_registry.load_model(
                deployment_config.model_name,
                deployment_config.version
            )
            
            # Dockerイメージの作成
            image_tag = self._build_docker_image(model, metadata)
            
            # デプロイメント戦略の実行
            if deployment_config.strategy == DeploymentStrategy.ROLLING:
                success = self._rolling_deployment(image_tag, deployment_config)
            elif deployment_config.strategy == DeploymentStrategy.BLUE_GREEN:
                success = self._blue_green_deployment(image_tag, deployment_config)
            elif deployment_config.strategy == DeploymentStrategy.CANARY:
                success = self._canary_deployment(image_tag, deployment_config)
            else:
                raise ValueError(f"Unknown deployment strategy: {deployment_config.strategy}")
            
            if success:
                # モデルステージの更新
                self.model_registry.transition_model_stage(
                    deployment_config.model_name,
                    deployment_config.version,
                    ModelStage.PRODUCTION
                )
            
            return success
        
        except Exception as e:
            self.logger.error(
                "Deployment failed",
                model_name=deployment_config.model_name,
                version=deployment_config.version,
                error=str(e)
            )
            return False
    
    def _build_docker_image(self, model: nn.Module,
                           metadata: Any) -> str:
        """Dockerイメージの作成"""
        try:
            # 一時ディレクトリの作成
            build_dir = Path("docker_build")
            build_dir.mkdir(exist_ok=True)
            
            # モデルの保存
            torch.save(model.state_dict(), build_dir / "model.pth")
            
            # メタデータの保存
            with open(build_dir / "metadata.json", 'w') as f:
                json.dump(metadata.__dict__, f)
            
            # Dockerfileの作成
            dockerfile_content = f"""
            FROM python:3.9-slim
            
            WORKDIR /app
            
            # 依存関係のインストール
            COPY requirements.txt .
            RUN pip install -r requirements.txt
            
            # モデルとメタデータのコピー
            COPY model.pth .
            COPY metadata.json .
            COPY serving.py .
            
            # Serving APIの起動
            CMD ["uvicorn", "serving:app", "--host", "0.0.0.0", "--port", "8000"]
            """
            
            with open(build_dir / "Dockerfile", 'w') as f:
                f.write(dockerfile_content)
            
            # Serving APIの作成
            serving_content = """
            from fastapi import FastAPI, HTTPException
            import torch
            import json
            
            app = FastAPI()
            
            # モデルの読み込み
            model = torch.load('model.pth')
            with open('metadata.json') as f:
                metadata = json.load(f)
            
            @app.post("/predict")
            async def predict(data: dict):
                try:
                    # 入力データの変換
                    input_tensor = torch.tensor(data['input'])
                    
                    # 推論の実行
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    return {"prediction": output.tolist()}
                
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @app.get("/health")
            async def health_check():
                return {"status": "healthy"}
            """
            
            with open(build_dir / "serving.py", 'w') as f:
                f.write(serving_content)
            
            # 依存関係の定義
            requirements_content = """
            torch==2.2.0
            fastapi==0.109.2
            uvicorn==0.27.1
            """
            
            with open(build_dir / "requirements.txt", 'w') as f:
                f.write(requirements_content)
            
            # イメージのビルド
            tag = f"{metadata.name}:{metadata.version}"
            self.docker_client.images.build(
                path=str(build_dir),
                tag=tag,
                rm=True
            )
            
            return tag
        
        finally:
            # 一時ディレクトリの削除
            import shutil
            shutil.rmtree(build_dir)
    
    def _rolling_deployment(self, image_tag: str,
                          config: DeploymentConfig) -> bool:
        """ローリングデプロイメント"""
        try:
            # デプロイメント定義
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(
                    name=f"{config.model_name}-deployment"
                ),
                spec=client.V1DeploymentSpec(
                    replicas=config.scaling_config.get('replicas', 3),
                    selector=client.V1LabelSelector(
                        match_labels={"app": config.model_name}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": config.model_name}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name=config.model_name,
                                    image=image_tag,
                                    ports=[
                                        client.V1ContainerPort(
                                            container_port=8000
                                        )
                                    ],
                                    resources=client.V1ResourceRequirements(
                                        requests=config.resource_requirements,
                                        limits={
                                            k: str(float(v) * 1.5)
                                            for k, v in config.resource_requirements.items()
                                        }
                                    ),
                                    liveness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path="/health",
                                            port=8000
                                        ),
                                        initial_delay_seconds=30,
                                        period_seconds=10
                                    ),
                                    readiness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path="/health",
                                            port=8000
                                        ),
                                        initial_delay_seconds=15,
                                        period_seconds=5
                                    )
                                )
                            ]
                        )
                    )
                )
            )
            
            # デプロイメントの作成または更新
            try:
                self.k8s_apps.create_namespaced_deployment(
                    namespace="default",
                    body=deployment
                )
            except kubernetes.client.rest.ApiException as e:
                if e.status == 409:  # Already exists
                    self.k8s_apps.replace_namespaced_deployment(
                        name=f"{config.model_name}-deployment",
                        namespace="default",
                        body=deployment
                    )
                else:
                    raise
            
            # デプロイメントの完了を待機
            self._wait_for_deployment(f"{config.model_name}-deployment")
            
            return True
        
        except Exception as e:
            self.logger.error(
                "Rolling deployment failed",
                model_name=config.model_name,
                error=str(e)
            )
            return False
    
    def _blue_green_deployment(self, image_tag: str,
                             config: DeploymentConfig) -> bool:
        """Blue-Greenデプロイメント"""
        try:
            # 現在のアクティブ環境の確認
            current_service = self.k8s_core.read_namespaced_service(
                name=f"{config.model_name}-service",
                namespace="default"
            )
            current_color = current_service.spec.selector.get("color", "blue")
            new_color = "green" if current_color == "blue" else "blue"
            
            # 新環境のデプロイメント
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(
                    name=f"{config.model_name}-{new_color}"
                ),
                spec=client.V1DeploymentSpec(
                    replicas=config.scaling_config.get('replicas', 3),
                    selector=client.V1LabelSelector(
                        match_labels={
                            "app": config.model_name,
                            "color": new_color
                        }
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={
                                "app": config.model_name,
                                "color": new_color
                            }
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name=config.model_name,
                                    image=image_tag,
                                    ports=[
                                        client.V1ContainerPort(
                                            container_port=8000
                                        )
                                    ],
                                    resources=client.V1ResourceRequirements(
                                        requests=config.resource_requirements,
                                        limits={
                                            k: str(float(v) * 1.5)
                                            for k, v in config.resource_requirements.items()
                                        }
                                    ),
                                    liveness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path="/health",
                                            port=8000
                                        ),
                                        initial_delay_seconds=30,
                                        period_seconds=10
                                    ),
                                    readiness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path="/health",
                                            port=8000
                                        ),
                                        initial_delay_seconds=15,
                                        period_seconds=5
                                    )
                                )
                            ]
                        )
                    )
                )
            )
            
            # 新環境のデプロイ
            self.k8s_apps.create_namespaced_deployment(
                namespace="default",
                body=deployment
            )
            
            # デプロイメントの完了を待機
            self._wait_for_deployment(f"{config.model_name}-{new_color}")
            
            # トラフィックの切り替え
            service = client.V1Service(
                metadata=client.V1ObjectMeta(
                    name=f"{config.model_name}-service"
                ),
                spec=client.V1ServiceSpec(
                    selector={
                        "app": config.model_name,
                        "color": new_color
                    },
                    ports=[
                        client.V1ServicePort(
                            port=80,
                            target_port=8000
                        )
                    ]
                )
            )
            
            self.k8s_core.replace_namespaced_service(
                name=f"{config.model_name}-service",
                namespace="default",
                body=service
            )
            
            # 古い環境の削除
            if config.rollback_config.get('keep_old_deployment', False):
                self.logger.info(
                    "Keeping old deployment for potential rollback",
                    old_color=current_color
                )
            else:
                self.k8s_apps.delete_namespaced_deployment(
                    name=f"{config.model_name}-{current_color}",
                    namespace="default"
                )
            
            return True
        
        except Exception as e:
            self.logger.error(
                "Blue-Green deployment failed",
                model_name=config.model_name,
                error=str(e)
            )
            return False
    
    def _canary_deployment(self, image_tag: str,
                          config: DeploymentConfig) -> bool:
        """Canaryデプロイメント"""
        try:
            # Canaryデプロイメントの作成
            canary_deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(
                    name=f"{config.model_name}-canary"
                ),
                spec=client.V1DeploymentSpec(
                    replicas=1,  # Canaryは小規模に開始
                    selector=client.V1LabelSelector(
                        match_labels={
                            "app": config.model_name,
                            "version": "canary"
                        }
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={
                                "app": config.model_name,
                                "version": "canary"
                            }
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name=config.model_name,
                                    image=image_tag,
                                    ports=[
                                        client.V1ContainerPort(
                                            container_port=8000
                                        )
                                    ],
                                    resources=client.V1ResourceRequirements(
                                        requests=config.resource_requirements,
                                        limits={
                                            k: str(float(v) * 1.5)
                                            for k, v in config.resource_requirements.items()
                                        }
                                    ),
                                    liveness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path="/health",
                                            port=8000
                                        ),
                                        initial_delay_seconds=30,
                                        period_seconds=10
                                    ),
                                    readiness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path="/health",
                                            port=8000
                                        ),
                                        initial_delay_seconds=15,
                                        period_seconds=5
                                    )
                                )
                            ]
                        )
                    )
                )
            )
            
            # Canaryデプロイメントの作成
            self.k8s_apps.create_namespaced_deployment(
                namespace="default",
                body=canary_deployment
            )
            
            # デプロイメントの完了を待機
            self._wait_for_deployment(f"{config.model_name}-canary")
            
            # トラフィックの段階的な移行
            for percentage in [25, 50, 75, 100]:
                # メトリクスの確認
                if not self._verify_canary_metrics():
                    self.logger.warning(
                        "Canary metrics check failed",
                        percentage=percentage
                    )
                    return False
                
                # トラフィック比率の更新
                self._update_traffic_split(
                    config.model_name,
                    percentage
                )
                
                # 安定化期間の待機
                import time
                time.sleep(300)  # 5分間の安定化期間
            
            # Canaryの昇格
            self._promote_canary(config.model_name)
            
            return True
        
        except Exception as e:
            self.logger.error(
                "Canary deployment failed",
                model_name=config.model_name,
                error=str(e)
            )
            return False
    
    def _wait_for_deployment(self, deployment_name: str):
        """デプロイメントの完了を待機"""
        while True:
            deployment = self.k8s_apps.read_namespaced_deployment(
                name=deployment_name,
                namespace="default"
            )
            
            if deployment.status.available_replicas == deployment.spec.replicas:
                break
            
            import time
            time.sleep(5)
    
    def _verify_canary_metrics(self) -> bool:
        """Canaryメトリクスの検証"""
        # メトリクスの検証ロジックをここに実装
        # 例: エラー率、レイテンシー、リソース使用率など
        return True
    
    def _update_traffic_split(self, model_name: str, canary_percentage: int):
        """トラフィック分割の更新"""
        # トラフィック分割の更新ロジックをここに実装
        pass
    
    def _promote_canary(self, model_name: str):
        """Canaryの昇格"""
        # Canaryを本番環境に昇格するロジックをここに実装
        pass
    
    def rollback_deployment(self, model_name: str, version: str) -> bool:
        """デプロイメントのロールバック"""
        try:
            # 前のバージョンの確認
            previous_version = self.model_registry.get_model_history(model_name)[-2]
            
            # ロールバック用の設定
            config = DeploymentConfig(
                model_name=model_name,
                version=previous_version['version'],
                strategy=DeploymentStrategy.ROLLING,
                resource_requirements={
                    "cpu": "500m",
                    "memory": "512Mi"
                },
                scaling_config={
                    "replicas": 3
                },
                health_check_config={
                    "initial_delay": 30,
                    "period": 10
                },
                rollback_config={
                    "keep_old_deployment": True
                }
            )
            
            # ロールバックの実行
            success = self.deploy_model(config)
            
            if success:
                # モデルステージの更新
                self.model_registry.transition_model_stage(
                    model_name,
                    version,
                    ModelStage.ARCHIVED
                )
                
                self.model_registry.transition_model_stage(
                    model_name,
                    previous_version['version'],
                    ModelStage.PRODUCTION
                )
            
            return success
        
        except Exception as e:
            self.logger.error(
                "Rollback failed",
                model_name=model_name,
                version=version,
                error=str(e)
            )
            return False
