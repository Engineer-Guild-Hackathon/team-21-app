import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
import json
from pathlib import Path
import structlog
from ..models.advanced_dqn import DuelingDQN
from ..models.emotion_analyzer import EmotionAnalyzer

class ModelType(Enum):
    """モデルタイプの定義"""
    DQN = "dqn"
    EMOTION = "emotion"
    FEEDBACK = "feedback"

@dataclass
class ClientConfig:
    """クライアント設定"""
    client_id: str
    model_type: ModelType
    local_epochs: int
    batch_size: int
    learning_rate: float
    max_grad_norm: float
    privacy_budget: float
    min_samples: int

@dataclass
class ModelUpdate:
    """モデル更新情報"""
    client_id: str
    model_type: ModelType
    parameters: Dict[str, torch.Tensor]
    metrics: Dict[str, float]
    training_size: int
    timestamp: float

class ClientLearner:
    """フェデレーテッドラーニングのクライアントサイド実装"""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # モデルの初期化
        self.model = self._initialize_model()
        self.optimizer = self._initialize_optimizer()
        
        # ロガーの設定
        self.logger = structlog.get_logger(__name__)
        
        # ローカルデータバッファ
        self.local_data_buffer = []
        self.local_data_size = 0
    
    def _initialize_model(self) -> nn.Module:
        """モデルの初期化"""
        if self.config.model_type == ModelType.DQN:
            return DuelingDQN(
                state_dim=8,
                action_dim=60,
                hidden_dim=256
            ).to(self.device)
        elif self.config.model_type == ModelType.EMOTION:
            return EmotionAnalyzer().bert_model.to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _initialize_optimizer(self) -> optim.Optimizer:
        """オプティマイザーの初期化"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
    
    def add_local_data(self, data: Dict[str, Any]):
        """ローカルデータの追加"""
        if self.local_data_size < self.config.min_samples:
            self.local_data_buffer.append(data)
            self.local_data_size += 1
            
            self.logger.info(
                "Added local data",
                current_size=self.local_data_size,
                min_samples=self.config.min_samples
            )
    
    def train_local_model(self) -> Optional[ModelUpdate]:
        """ローカルモデルの学習"""
        if self.local_data_size < self.config.min_samples:
            self.logger.warning(
                "Insufficient local data for training",
                current_size=self.local_data_size,
                min_samples=self.config.min_samples
            )
            return None
        
        try:
            # データの準備
            train_data = self._prepare_training_data()
            
            # 学習ループ
            metrics = self._train_epochs(train_data)
            
            # モデル更新の作成
            update = self._create_model_update(metrics)
            
            # プライバシー保護の適用
            protected_update = self._apply_privacy_protection(update)
            
            self.logger.info(
                "Local training completed",
                metrics=metrics
            )
            
            return protected_update
        
        except Exception as e:
            self.logger.error(
                "Failed to train local model",
                error=str(e)
            )
            raise
    
    def _prepare_training_data(self) -> List[Dict[str, torch.Tensor]]:
        """学習データの準備"""
        if self.config.model_type == ModelType.DQN:
            return self._prepare_dqn_data()
        elif self.config.model_type == ModelType.EMOTION:
            return self._prepare_emotion_data()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _prepare_dqn_data(self) -> List[Dict[str, torch.Tensor]]:
        """DQN用データの準備"""
        prepared_data = []
        for data in self.local_data_buffer:
            # 状態の変換
            state = torch.FloatTensor(data['state']).to(self.device)
            next_state = torch.FloatTensor(data['next_state']).to(self.device)
            action = torch.LongTensor([data['action']]).to(self.device)
            reward = torch.FloatTensor([data['reward']]).to(self.device)
            done = torch.BoolTensor([data['done']]).to(self.device)
            
            prepared_data.append({
                'state': state,
                'next_state': next_state,
                'action': action,
                'reward': reward,
                'done': done
            })
        
        return prepared_data
    
    def _prepare_emotion_data(self) -> List[Dict[str, torch.Tensor]]:
        """感情分析用データの準備"""
        prepared_data = []
        for data in self.local_data_buffer:
            # テキストのトークン化
            tokens = self.model.tokenizer(
                data['text'],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            label = torch.LongTensor([data['label']]).to(self.device)
            
            prepared_data.append({
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
                'label': label
            })
        
        return prepared_data
    
    def _train_epochs(self, train_data: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """エポック単位の学習"""
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0
        }
        
        self.model.train()
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            # バッチ処理
            for i in range(0, len(train_data), self.config.batch_size):
                batch = train_data[i:i + self.config.batch_size]
                
                # バッチデータの結合
                if self.config.model_type == ModelType.DQN:
                    loss, acc = self._train_dqn_batch(batch)
                else:
                    loss, acc = self._train_emotion_batch(batch)
                
                epoch_loss += loss
                correct += acc[0]
                total += acc[1]
            
            # エポックメトリクスの計算
            metrics['loss'] = epoch_loss / len(train_data)
            metrics['accuracy'] = correct / total if total > 0 else 0.0
            
            self.logger.info(
                "Epoch completed",
                epoch=epoch + 1,
                loss=metrics['loss'],
                accuracy=metrics['accuracy']
            )
        
        return metrics
    
    def _train_dqn_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Tuple[float, Tuple[int, int]]:
        """DQNバッチの学習"""
        self.optimizer.zero_grad()
        
        # バッチデータの結合
        states = torch.cat([b['state'] for b in batch])
        next_states = torch.cat([b['next_state'] for b in batch])
        actions = torch.cat([b['action'] for b in batch])
        rewards = torch.cat([b['reward'] for b in batch])
        dones = torch.cat([b['done'] for b in batch])
        
        # Q値の計算
        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones.float()) * 0.99 * next_q_values
        
        # 損失の計算
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        # 勾配の計算と更新
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        # 精度の計算（Q値の予測精度）
        predicted = (current_q_values > expected_q_values.unsqueeze(1)).float()
        correct = (predicted == (rewards > 0).float().unsqueeze(1)).sum().item()
        
        return loss.item(), (correct, len(batch))
    
    def _train_emotion_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Tuple[float, Tuple[int, int]]:
        """感情分析バッチの学習"""
        self.optimizer.zero_grad()
        
        # バッチデータの結合
        input_ids = torch.cat([b['input_ids'] for b in batch])
        attention_mask = torch.cat([b['attention_mask'] for b in batch])
        labels = torch.cat([b['label'] for b in batch])
        
        # モデル出力の計算
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # 勾配の計算と更新
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        # 精度の計算
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).sum().item()
        
        return loss.item(), (correct, len(batch))
    
    def _create_model_update(self, metrics: Dict[str, float]) -> ModelUpdate:
        """モデル更新の作成"""
        parameters = {
            name: param.data.cpu()
            for name, param in self.model.named_parameters()
        }
        
        return ModelUpdate(
            client_id=self.config.client_id,
            model_type=self.config.model_type,
            parameters=parameters,
            metrics=metrics,
            training_size=self.local_data_size,
            timestamp=time.time()
        )
    
    def _apply_privacy_protection(self, update: ModelUpdate) -> ModelUpdate:
        """プライバシー保護の適用"""
        # 差分プライバシーの適用
        epsilon = self.config.privacy_budget
        
        for name, param in update.parameters.items():
            # ガウシアンノイズの追加
            noise = torch.randn_like(param) * (1.0 / epsilon)
            param.add_(noise)
        
        return update
    
    def update_local_model(self, global_parameters: Dict[str, torch.Tensor]):
        """グローバルモデルの更新の適用"""
        try:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in global_parameters:
                        param.data.copy_(global_parameters[name])
            
            self.logger.info("Applied global model update")
        
        except Exception as e:
            self.logger.error(
                "Failed to apply global model update",
                error=str(e)
            )
            raise
    
    def save_local_model(self, path: str):
        """ローカルモデルの保存"""
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config.__dict__
            }, save_path)
            
            self.logger.info("Saved local model", path=str(save_path))
        
        except Exception as e:
            self.logger.error(
                "Failed to save local model",
                path=str(save_path),
                error=str(e)
            )
            raise
    
    def load_local_model(self, path: str):
        """ローカルモデルの読み込み"""
        try:
            checkpoint = torch.load(path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.logger.info("Loaded local model", path=path)
        
        except Exception as e:
            self.logger.error(
                "Failed to load local model",
                path=path,
                error=str(e)
            )
            raise
