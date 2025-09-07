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
import aiohttp
from aiohttp import web
import zmq.asyncio
import pickle
from .client_learner import ClientConfig, ModelType
from .model_aggregator import AggregationConfig
from .privacy_mechanism import PrivacyConfig

@dataclass
class OrchestratorConfig:
    """オーケストレーター設定"""
    min_clients: int
    max_clients: int
    round_timeout: float
    aggregation_interval: float
    checkpoint_interval: int
    max_rounds: int
    early_stopping_patience: int
    early_stopping_threshold: float

class FederatedOrchestrator:
    """分散学習オーケストレーター"""
    
    def __init__(self, config: OrchestratorConfig,
                 client_config: ClientConfig,
                 aggregation_config: AggregationConfig,
                 privacy_config: PrivacyConfig):
        self.config = config
        self.client_config = client_config
        self.aggregation_config = aggregation_config
        self.privacy_config = privacy_config
        
        # クライアント管理
        self.active_clients = {}
        self.client_metrics = {}
        
        # 学習状態
        self.current_round = 0
        self.best_metrics = {
            'loss': float('inf'),
            'accuracy': 0.0,
            'round': 0
        }
        
        # 早期停止
        self.patience_counter = 0
        
        # ZMQコンテキスト
        self.context = zmq.asyncio.Context()
        
        # ロガーの設定
        self.logger = structlog.get_logger(__name__)
    
    async def start_server(self, host: str = 'localhost', port: int = 5555):
        """サーバーの起動"""
        try:
            # ZMQソケットの設定
            self.socket = self.context.socket(zmq.ROUTER)
            self.socket.bind(f"tcp://{host}:{port}")
            
            self.logger.info(
                "Server started",
                host=host,
                port=port
            )
            
            # メインループの開始
            await self._main_loop()
        
        except Exception as e:
            self.logger.error(
                "Failed to start server",
                error=str(e)
            )
            raise
        
        finally:
            self.socket.close()
            self.context.term()
    
    async def _main_loop(self):
        """メインループ"""
        while self.current_round < self.config.max_rounds:
            try:
                # ラウンドの開始
                self.logger.info(
                    "Starting training round",
                    round=self.current_round + 1
                )
                
                # クライアントの選択
                selected_clients = await self._select_clients()
                
                if len(selected_clients) < self.config.min_clients:
                    self.logger.warning(
                        "Insufficient clients for training",
                        active=len(selected_clients),
                        required=self.config.min_clients
                    )
                    continue
                
                # トレーニングラウンドの実行
                success = await self._execute_training_round(selected_clients)
                
                if success:
                    self.current_round += 1
                    
                    # チェックポイントの保存
                    if self.current_round % self.config.checkpoint_interval == 0:
                        await self._save_checkpoint()
                    
                    # 早期停止の確認
                    if await self._check_early_stopping():
                        self.logger.info("Early stopping triggered")
                        break
                
            except Exception as e:
                self.logger.error(
                    "Error in training round",
                    round=self.current_round + 1,
                    error=str(e)
                )
                continue
    
    async def _select_clients(self) -> List[str]:
        """参加クライアントの選択"""
        # アクティブクライアントの確認
        active_clients = list(self.active_clients.keys())
        
        if len(active_clients) > self.config.max_clients:
            # クライアントの性能に基づく選択
            client_scores = []
            for client_id in active_clients:
                metrics = self.client_metrics.get(client_id, {})
                score = metrics.get('accuracy', 0.0) * 0.7 + \
                        (1.0 - metrics.get('loss', float('inf')) / 10.0) * 0.3
                client_scores.append((client_id, score))
            
            # スコアでソートして上位を選択
            client_scores.sort(key=lambda x: x[1], reverse=True)
            selected_clients = [c[0] for c in client_scores[:self.config.max_clients]]
        else:
            selected_clients = active_clients
        
        return selected_clients
    
    async def _execute_training_round(self, clients: List[str]) -> bool:
        """トレーニングラウンドの実行"""
        try:
            # トレーニング開始の通知
            for client_id in clients:
                await self._send_message(
                    client_id,
                    {
                        'type': 'start_training',
                        'round': self.current_round + 1
                    }
                )
            
            # 更新の収集
            updates = []
            async with asyncio.timeout(self.config.round_timeout):
                for client_id in clients:
                    update = await self._receive_update(client_id)
                    if update:
                        updates.append(update)
            
            if len(updates) < self.config.min_clients:
                self.logger.warning(
                    "Insufficient updates received",
                    received=len(updates),
                    required=self.config.min_clients
                )
                return False
            
            # 更新の集約
            aggregated_model = await self._aggregate_updates(updates)
            
            # 新しいモデルの配布
            await self._distribute_model(aggregated_model, clients)
            
            # メトリクスの更新
            await self._update_metrics(updates)
            
            return True
        
        except asyncio.TimeoutError:
            self.logger.error("Round timeout")
            return False
        
        except Exception as e:
            self.logger.error(
                "Failed to execute training round",
                error=str(e)
            )
            return False
    
    async def _send_message(self, client_id: str, message: Dict[str, Any]):
        """メッセージの送信"""
        try:
            serialized_message = pickle.dumps(message)
            await self.socket.send_multipart([
                client_id.encode(),
                serialized_message
            ])
        
        except Exception as e:
            self.logger.error(
                "Failed to send message",
                client_id=client_id,
                error=str(e)
            )
            raise
    
    async def _receive_update(self, client_id: str) -> Optional[Dict[str, Any]]:
        """更新の受信"""
        try:
            identity, message = await self.socket.recv_multipart()
            update = pickle.loads(message)
            
            if update['client_id'] != client_id:
                self.logger.warning(
                    "Received update from unexpected client",
                    expected=client_id,
                    received=update['client_id']
                )
                return None
            
            return update
        
        except Exception as e:
            self.logger.error(
                "Failed to receive update",
                client_id=client_id,
                error=str(e)
            )
            return None
    
    async def _aggregate_updates(self, updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """更新の集約"""
        try:
            # プライバシー保護の適用
            protected_updates = []
            for update in updates:
                protected_update = await self._apply_privacy_protection(update)
                protected_updates.append(protected_update)
            
            # セキュア集約の実行
            aggregated_model = await self._secure_aggregate(protected_updates)
            
            return aggregated_model
        
        except Exception as e:
            self.logger.error(
                "Failed to aggregate updates",
                error=str(e)
            )
            raise
    
    async def _apply_privacy_protection(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """プライバシー保護の適用"""
        try:
            # 差分プライバシーの適用
            protected_parameters = self.privacy_mechanism.apply_differential_privacy(
                update['parameters']
            )
            
            # パラメータの暗号化
            encrypted_parameters = self.privacy_mechanism.encrypt_parameters(
                protected_parameters
            )
            
            return {
                'client_id': update['client_id'],
                'parameters': encrypted_parameters,
                'metrics': update['metrics']
            }
        
        except Exception as e:
            self.logger.error(
                "Failed to apply privacy protection",
                client_id=update['client_id'],
                error=str(e)
            )
            raise
    
    async def _secure_aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """セキュア集約の実行"""
        try:
            # パラメータの復号化
            decrypted_updates = []
            for update in updates:
                parameters = self.privacy_mechanism.decrypt_parameters(
                    update['parameters']
                )
                decrypted_updates.append(parameters)
            
            # セキュア集約の実行
            aggregated_model = self.privacy_mechanism.secure_aggregate(
                decrypted_updates
            )
            
            return aggregated_model
        
        except Exception as e:
            self.logger.error(
                "Failed to perform secure aggregation",
                error=str(e)
            )
            raise
    
    async def _distribute_model(self, model: Dict[str, torch.Tensor],
                              clients: List[str]):
        """モデルの配布"""
        try:
            # モデルの暗号化
            encrypted_model = self.privacy_mechanism.encrypt_parameters(model)
            
            # 各クライアントへの送信
            for client_id in clients:
                await self._send_message(
                    client_id,
                    {
                        'type': 'model_update',
                        'parameters': encrypted_model
                    }
                )
        
        except Exception as e:
            self.logger.error(
                "Failed to distribute model",
                error=str(e)
            )
            raise
    
    async def _update_metrics(self, updates: List[Dict[str, Any]]):
        """メトリクスの更新"""
        try:
            # クライアントメトリクスの更新
            for update in updates:
                self.client_metrics[update['client_id']] = update['metrics']
            
            # グローバルメトリクスの計算
            total_samples = sum(u['training_size'] for u in updates)
            weighted_loss = sum(
                u['metrics']['loss'] * u['training_size'] / total_samples
                for u in updates
            )
            weighted_accuracy = sum(
                u['metrics']['accuracy'] * u['training_size'] / total_samples
                for u in updates
            )
            
            # ベストメトリクスの更新
            if weighted_accuracy > self.best_metrics['accuracy']:
                self.best_metrics = {
                    'loss': weighted_loss,
                    'accuracy': weighted_accuracy,
                    'round': self.current_round + 1
                }
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        except Exception as e:
            self.logger.error(
                "Failed to update metrics",
                error=str(e)
            )
            raise
    
    async def _check_early_stopping(self) -> bool:
        """早期停止の確認"""
        if self.patience_counter >= self.config.early_stopping_patience:
            current_accuracy = self.client_metrics['accuracy']
            best_accuracy = self.best_metrics['accuracy']
            
            if (best_accuracy - current_accuracy) > self.config.early_stopping_threshold:
                return True
        
        return False
    
    async def _save_checkpoint(self):
        """チェックポイントの保存"""
        try:
            checkpoint = {
                'round': self.current_round,
                'best_metrics': self.best_metrics,
                'client_metrics': self.client_metrics,
                'model_state': self.global_model,
                'timestamp': datetime.now().isoformat()
            }
            
            save_path = Path(f"checkpoints/round_{self.current_round}.pt")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(checkpoint, save_path)
            
            self.logger.info(
                "Saved checkpoint",
                round=self.current_round,
                path=str(save_path)
            )
        
        except Exception as e:
            self.logger.error(
                "Failed to save checkpoint",
                round=self.current_round,
                error=str(e)
            )
            raise
    
    async def handle_client_connection(self, client_id: str, message: Dict[str, Any]):
        """クライアント接続の処理"""
        try:
            message_type = message.get('type')
            
            if message_type == 'register':
                # クライアントの登録
                self.active_clients[client_id] = {
                    'status': 'active',
                    'last_seen': datetime.now()
                }
                
                await self._send_message(
                    client_id,
                    {
                        'type': 'registration_success',
                        'config': self.client_config.__dict__
                    }
                )
            
            elif message_type == 'heartbeat':
                # ハートビートの更新
                if client_id in self.active_clients:
                    self.active_clients[client_id]['last_seen'] = datetime.now()
            
            elif message_type == 'disconnect':
                # クライアントの切断
                if client_id in self.active_clients:
                    del self.active_clients[client_id]
            
            else:
                self.logger.warning(
                    "Unknown message type",
                    client_id=client_id,
                    type=message_type
                )
        
        except Exception as e:
            self.logger.error(
                "Failed to handle client connection",
                client_id=client_id,
                error=str(e)
            )
            raise
    
    async def monitor_clients(self):
        """クライアントの監視"""
        while True:
            try:
                current_time = datetime.now()
                
                # タイムアウトしたクライアントの検出
                timeout_threshold = timedelta(seconds=self.config.round_timeout * 2)
                disconnected_clients = []
                
                for client_id, info in self.active_clients.items():
                    if (current_time - info['last_seen']) > timeout_threshold:
                        disconnected_clients.append(client_id)
                
                # タイムアウトしたクライアントの削除
                for client_id in disconnected_clients:
                    del self.active_clients[client_id]
                    self.logger.warning(
                        "Client disconnected due to timeout",
                        client_id=client_id
                    )
                
                await asyncio.sleep(60)  # 1分ごとに確認
            
            except Exception as e:
                self.logger.error(
                    "Error in client monitoring",
                    error=str(e)
                )
                await asyncio.sleep(60)  # エラー時も1分待機
