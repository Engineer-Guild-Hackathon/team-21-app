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
import asyncio
from .client_learner import ModelType, ModelUpdate

@dataclass
class AggregationConfig:
    """集約設定"""
    min_clients: int
    max_wait_time: float
    aggregation_method: str
    weight_method: str
    privacy_threshold: float
    update_threshold: float

class ModelAggregator:
    """フェデレーテッドラーニングのモデル集約システム"""
    
    def __init__(self, config: AggregationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # グローバルモデルの状態
        self.global_models = {}
        self.round_updates = {}
        self.client_weights = {}
        
        # 集約メトリクス
        self.aggregation_metrics = {
            'round': 0,
            'participating_clients': 0,
            'total_samples': 0,
            'average_loss': 0.0,
            'average_accuracy': 0.0
        }
        
        # ロガーの設定
        self.logger = structlog.get_logger(__name__)
    
    def initialize_global_model(self, model_type: ModelType,
                              initial_parameters: Dict[str, torch.Tensor]):
        """グローバルモデルの初期化"""
        self.global_models[model_type] = {
            name: param.clone().to(self.device)
            for name, param in initial_parameters.items()
        }
        
        self.round_updates[model_type] = []
        
        self.logger.info(
            "Initialized global model",
            model_type=model_type.value
        )
    
    async def aggregate_updates(self, model_type: ModelType) -> Optional[Dict[str, torch.Tensor]]:
        """モデル更新の集約"""
        updates = self.round_updates[model_type]
        
        if len(updates) < self.config.min_clients:
            self.logger.warning(
                "Insufficient clients for aggregation",
                current_clients=len(updates),
                min_clients=self.config.min_clients
            )
            return None
        
        try:
            # プライバシー検証
            if not self._verify_privacy(updates):
                self.logger.error("Privacy verification failed")
                return None
            
            # 更新の検証
            if not self._verify_updates(updates):
                self.logger.error("Update verification failed")
                return None
            
            # 重みの計算
            weights = self._compute_weights(updates)
            
            # モデルの集約
            aggregated_model = self._aggregate_models(updates, weights)
            
            # メトリクスの更新
            self._update_metrics(updates)
            
            self.logger.info(
                "Model aggregation completed",
                round=self.aggregation_metrics['round'],
                clients=len(updates),
                samples=self.aggregation_metrics['total_samples']
            )
            
            # ラウンドの更新をクリア
            self.round_updates[model_type] = []
            
            return aggregated_model
        
        except Exception as e:
            self.logger.error(
                "Failed to aggregate models",
                error=str(e)
            )
            raise
    
    def add_client_update(self, update: ModelUpdate):
        """クライアントの更新を追加"""
        try:
            # 更新の基本検証
            if not self._validate_update(update):
                self.logger.warning(
                    "Invalid client update",
                    client_id=update.client_id
                )
                return
            
            # 更新の追加
            self.round_updates[update.model_type].append(update)
            
            self.logger.info(
                "Added client update",
                client_id=update.client_id,
                model_type=update.model_type.value,
                metrics=update.metrics
            )
        
        except Exception as e:
            self.logger.error(
                "Failed to add client update",
                client_id=update.client_id,
                error=str(e)
            )
            raise
    
    def _validate_update(self, update: ModelUpdate) -> bool:
        """更新の検証"""
        # モデルタイプの確認
        if update.model_type not in self.global_models:
            self.logger.error(
                "Unknown model type",
                model_type=update.model_type.value
            )
            return False
        
        # パラメータ構造の確認
        global_params = self.global_models[update.model_type]
        if set(update.parameters.keys()) != set(global_params.keys()):
            self.logger.error(
                "Parameter structure mismatch",
                client_id=update.client_id
            )
            return False
        
        # メトリクスの確認
        required_metrics = {'loss', 'accuracy'}
        if not required_metrics.issubset(update.metrics.keys()):
            self.logger.error(
                "Missing required metrics",
                client_id=update.client_id
            )
            return False
        
        return True
    
    def _verify_privacy(self, updates: List[ModelUpdate]) -> bool:
        """プライバシー要件の検証"""
        for update in updates:
            # パラメータの変化量を計算
            global_params = self.global_models[update.model_type]
            param_changes = []
            
            for name, param in update.parameters.items():
                change = torch.norm(param - global_params[name]).item()
                param_changes.append(change)
            
            # 最大変化量の確認
            max_change = max(param_changes)
            if max_change > self.config.privacy_threshold:
                self.logger.warning(
                    "Privacy threshold exceeded",
                    client_id=update.client_id,
                    max_change=max_change
                )
                return False
        
        return True
    
    def _verify_updates(self, updates: List[ModelUpdate]) -> bool:
        """更新の妥当性検証"""
        for update in updates:
            # パラメータの変化率を計算
            global_params = self.global_models[update.model_type]
            relative_changes = []
            
            for name, param in update.parameters.items():
                relative_change = torch.norm(param - global_params[name]) / \
                                torch.norm(global_params[name])
                relative_changes.append(relative_change.item())
            
            # 最大変化率の確認
            max_change = max(relative_changes)
            if max_change > self.config.update_threshold:
                self.logger.warning(
                    "Update threshold exceeded",
                    client_id=update.client_id,
                    max_change=max_change
                )
                return False
        
        return True
    
    def _compute_weights(self, updates: List[ModelUpdate]) -> Dict[str, float]:
        """クライアントの重みを計算"""
        weights = {}
        
        if self.config.weight_method == "uniform":
            # 均一な重み
            weight = 1.0 / len(updates)
            weights = {update.client_id: weight for update in updates}
            
        elif self.config.weight_method == "sample_size":
            # データ量に基づく重み
            total_samples = sum(update.training_size for update in updates)
            weights = {
                update.client_id: update.training_size / total_samples
                for update in updates
            }
            
        elif self.config.weight_method == "performance":
            # 性能に基づく重み
            total_performance = sum(
                1.0 / (update.metrics['loss'] + 1e-8)
                for update in updates
            )
            weights = {
                update.client_id: (1.0 / (update.metrics['loss'] + 1e-8)) / total_performance
                for update in updates
            }
        
        return weights
    
    def _aggregate_models(self, updates: List[ModelUpdate],
                         weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """モデルの集約"""
        aggregated_params = {}
        
        if self.config.aggregation_method == "fedavg":
            # Federated Averaging
            for name in updates[0].parameters.keys():
                weighted_sum = torch.zeros_like(
                    updates[0].parameters[name],
                    device=self.device
                )
                
                for update in updates:
                    client_weight = weights[update.client_id]
                    weighted_sum.add_(
                        update.parameters[name].to(self.device) * client_weight
                    )
                
                aggregated_params[name] = weighted_sum
            
        elif self.config.aggregation_method == "median":
            # 中央値ベースの集約
            for name in updates[0].parameters.keys():
                stacked_params = torch.stack([
                    update.parameters[name].to(self.device)
                    for update in updates
                ])
                aggregated_params[name] = torch.median(stacked_params, dim=0)[0]
            
        elif self.config.aggregation_method == "trimmed_mean":
            # トリム平均による集約
            trim_ratio = 0.1  # 上下10%をトリム
            for name in updates[0].parameters.keys():
                stacked_params = torch.stack([
                    update.parameters[name].to(self.device)
                    for update in updates
                ])
                sorted_params, _ = torch.sort(stacked_params, dim=0)
                trim_size = int(len(updates) * trim_ratio)
                trimmed_params = sorted_params[trim_size:-trim_size]
                aggregated_params[name] = torch.mean(trimmed_params, dim=0)
        
        return aggregated_params
    
    def _update_metrics(self, updates: List[ModelUpdate]):
        """集約メトリクスの更新"""
        self.aggregation_metrics['round'] += 1
        self.aggregation_metrics['participating_clients'] = len(updates)
        self.aggregation_metrics['total_samples'] = sum(
            update.training_size for update in updates
        )
        self.aggregation_metrics['average_loss'] = np.mean([
            update.metrics['loss'] for update in updates
        ])
        self.aggregation_metrics['average_accuracy'] = np.mean([
            update.metrics['accuracy'] for update in updates
        ])
    
    def save_global_model(self, model_type: ModelType, path: str):
        """グローバルモデルの保存"""
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_type': model_type.value,
                'parameters': self.global_models[model_type],
                'metrics': self.aggregation_metrics,
                'timestamp': datetime.now().isoformat()
            }, save_path)
            
            self.logger.info(
                "Saved global model",
                model_type=model_type.value,
                path=str(save_path)
            )
        
        except Exception as e:
            self.logger.error(
                "Failed to save global model",
                model_type=model_type.value,
                path=str(save_path),
                error=str(e)
            )
            raise
    
    def load_global_model(self, path: str) -> ModelType:
        """グローバルモデルの読み込み"""
        try:
            checkpoint = torch.load(path)
            
            model_type = ModelType(checkpoint['model_type'])
            self.global_models[model_type] = {
                name: param.to(self.device)
                for name, param in checkpoint['parameters'].items()
            }
            
            self.aggregation_metrics = checkpoint['metrics']
            
            self.logger.info(
                "Loaded global model",
                model_type=model_type.value,
                path=path
            )
            
            return model_type
        
        except Exception as e:
            self.logger.error(
                "Failed to load global model",
                path=path,
                error=str(e)
            )
            raise
