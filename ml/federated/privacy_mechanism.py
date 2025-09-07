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
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

@dataclass
class PrivacyConfig:
    """プライバシー設定"""
    epsilon: float  # プライバシー予算
    delta: float   # プライバシー緩和パラメータ
    noise_multiplier: float  # ノイズ乗数
    max_grad_norm: float    # 勾配クリッピング閾値
    secure_aggregation: bool  # セキュア集約の有効化
    encryption_method: str   # 暗号化方式

class PrivacyMechanism:
    """プライバシー保護メカニズム"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 暗号化キーの生成
        self.encryption_key = self._generate_encryption_key()
        
        # プライバシー予算の追跡
        self.privacy_accountant = {
            'spent_budget': 0.0,
            'remaining_budget': config.epsilon
        }
        
        # ロガーの設定
        self.logger = structlog.get_logger(__name__)
    
    def _generate_encryption_key(self) -> bytes:
        """暗号化キーの生成"""
        if self.config.encryption_method == "fernet":
            return Fernet.generate_key()
        elif self.config.encryption_method == "rsa":
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            return private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            raise ValueError(f"Unsupported encryption method: {self.config.encryption_method}")
    
    def apply_differential_privacy(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """差分プライバシーの適用"""
        try:
            # プライバシー予算の確認
            if self.privacy_accountant['remaining_budget'] <= 0:
                raise ValueError("Privacy budget exhausted")
            
            # パラメータのクリッピング
            clipped_params = self._clip_parameters(parameters)
            
            # ノイズの追加
            noised_params = self._add_noise(clipped_params)
            
            # プライバシー予算の更新
            privacy_cost = self._calculate_privacy_cost(parameters)
            self._update_privacy_budget(privacy_cost)
            
            return noised_params
        
        except Exception as e:
            self.logger.error(
                "Failed to apply differential privacy",
                error=str(e)
            )
            raise
    
    def _clip_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """パラメータのクリッピング"""
        clipped_params = {}
        
        for name, param in parameters.items():
            # L2ノルムの計算
            param_norm = torch.norm(param)
            
            # クリッピング係数の計算
            clip_coef = min(1.0, self.config.max_grad_norm / (param_norm + 1e-6))
            
            # パラメータのクリッピング
            clipped_params[name] = param * clip_coef
        
        return clipped_params
    
    def _add_noise(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ガウシアンノイズの追加"""
        noised_params = {}
        
        for name, param in parameters.items():
            # ノイズのスケールを計算
            noise_scale = self.config.noise_multiplier * self.config.max_grad_norm
            
            # ガウシアンノイズの生成と追加
            noise = torch.randn_like(param) * noise_scale
            noised_params[name] = param + noise
        
        return noised_params
    
    def _calculate_privacy_cost(self, parameters: Dict[str, torch.Tensor]) -> float:
        """プライバシーコストの計算"""
        # モーメントアカウンタントによるプライバシー損失の計算
        q = 0.01  # サンプリング確率
        steps = 1000  # 学習ステップ数
        
        # RDPアカウンタントによるεの計算
        orders = [1 + x / 10.0 for x in range(1, 100)]
        rdp = self._compute_rdp(q, self.config.noise_multiplier, steps, orders)
        privacy_loss = self._get_privacy_spent(orders, rdp, target_delta=self.config.delta)
        
        return privacy_loss
    
    def _compute_rdp(self, q: float, noise_multiplier: float,
                     steps: int, orders: List[float]) -> List[float]:
        """RDPの計算"""
        # サブサンプリングを考慮したRDPの計算
        rdp = []
        for alpha in orders:
            term1 = q * q * alpha / (2 * (noise_multiplier ** 2))
            term2 = q * q * (alpha - 1) / (2 * (noise_multiplier ** 2))
            rdp.append(steps * (term1 + term2))
        
        return rdp
    
    def _get_privacy_spent(self, orders: List[float], rdp: List[float],
                          target_delta: float) -> float:
        """プライバシー損失の計算"""
        # 最適なεの探索
        eps = float('inf')
        
        for alpha, curr_rdp in zip(orders, rdp):
            if curr_rdp < float('inf'):
                current_eps = curr_rdp + (np.log(1 / target_delta) + np.log(alpha)) / (alpha - 1)
                eps = min(eps, current_eps)
        
        return eps
    
    def _update_privacy_budget(self, privacy_cost: float):
        """プライバシー予算の更新"""
        self.privacy_accountant['spent_budget'] += privacy_cost
        self.privacy_accountant['remaining_budget'] = max(
            0.0,
            self.config.epsilon - self.privacy_accountant['spent_budget']
        )
        
        self.logger.info(
            "Updated privacy budget",
            spent=self.privacy_accountant['spent_budget'],
            remaining=self.privacy_accountant['remaining_budget']
        )
    
    def encrypt_parameters(self, parameters: Dict[str, torch.Tensor]) -> bytes:
        """パラメータの暗号化"""
        try:
            # パラメータのシリアライズ
            serialized_params = {
                name: param.cpu().numpy().tobytes()
                for name, param in parameters.items()
            }
            
            if self.config.encryption_method == "fernet":
                return self._encrypt_fernet(serialized_params)
            elif self.config.encryption_method == "rsa":
                return self._encrypt_rsa(serialized_params)
            else:
                raise ValueError(f"Unsupported encryption method: {self.config.encryption_method}")
        
        except Exception as e:
            self.logger.error(
                "Failed to encrypt parameters",
                error=str(e)
            )
            raise
    
    def decrypt_parameters(self, encrypted_data: bytes) -> Dict[str, torch.Tensor]:
        """パラメータの復号化"""
        try:
            if self.config.encryption_method == "fernet":
                serialized_params = self._decrypt_fernet(encrypted_data)
            elif self.config.encryption_method == "rsa":
                serialized_params = self._decrypt_rsa(encrypted_data)
            else:
                raise ValueError(f"Unsupported encryption method: {self.config.encryption_method}")
            
            # パラメータの復元
            decrypted_params = {
                name: torch.from_numpy(
                    np.frombuffer(param_bytes, dtype=np.float32)
                ).to(self.device)
                for name, param_bytes in serialized_params.items()
            }
            
            return decrypted_params
        
        except Exception as e:
            self.logger.error(
                "Failed to decrypt parameters",
                error=str(e)
            )
            raise
    
    def _encrypt_fernet(self, data: Dict[str, bytes]) -> bytes:
        """Fernetによる暗号化"""
        f = Fernet(self.encryption_key)
        serialized_data = json.dumps(data).encode()
        return f.encrypt(serialized_data)
    
    def _decrypt_fernet(self, encrypted_data: bytes) -> Dict[str, bytes]:
        """Fernetによる復号化"""
        f = Fernet(self.encryption_key)
        decrypted_data = f.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
    
    def _encrypt_rsa(self, data: Dict[str, bytes]) -> bytes:
        """RSAによる暗号化"""
        public_key = serialization.load_pem_private_key(
            self.encryption_key,
            password=None
        ).public_key()
        
        encrypted_data = {}
        for name, value in data.items():
            encrypted_value = public_key.encrypt(
                value,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            encrypted_data[name] = base64.b64encode(encrypted_value).decode()
        
        return json.dumps(encrypted_data).encode()
    
    def _decrypt_rsa(self, encrypted_data: bytes) -> Dict[str, bytes]:
        """RSAによる復号化"""
        private_key = serialization.load_pem_private_key(
            self.encryption_key,
            password=None
        )
        
        encrypted_dict = json.loads(encrypted_data.decode())
        decrypted_data = {}
        
        for name, value in encrypted_dict.items():
            encrypted_value = base64.b64decode(value)
            decrypted_value = private_key.decrypt(
                encrypted_value,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            decrypted_data[name] = decrypted_value
        
        return decrypted_data
    
    def secure_aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """セキュア集約の実行"""
        if not self.config.secure_aggregation:
            return self._simple_average(updates)
        
        try:
            # マスク値の生成
            masks = self._generate_masks(len(updates))
            
            # マスク適用
            masked_updates = []
            for i, update in enumerate(updates):
                masked_update = {}
                for name, param in update.items():
                    masked_update[name] = param + masks[i][name]
                masked_updates.append(masked_update)
            
            # マスク済み更新の集約
            aggregated = self._simple_average(masked_updates)
            
            # マスクの除去
            final_update = {}
            for name in aggregated.keys():
                mask_sum = sum(mask[name] for mask in masks)
                final_update[name] = aggregated[name] - mask_sum / len(updates)
            
            return final_update
        
        except Exception as e:
            self.logger.error(
                "Failed to perform secure aggregation",
                error=str(e)
            )
            raise
    
    def _generate_masks(self, num_clients: int) -> List[Dict[str, torch.Tensor]]:
        """マスク値の生成"""
        masks = []
        for _ in range(num_clients):
            client_masks = {}
            for name, param in self.model_structure.items():
                mask = torch.randn_like(param)
                client_masks[name] = mask
            masks.append(client_masks)
        
        return masks
    
    def _simple_average(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """単純平均の計算"""
        averaged = {}
        for name in updates[0].keys():
            stacked = torch.stack([u[name] for u in updates])
            averaged[name] = torch.mean(stacked, dim=0)
        return averaged
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """プライバシーレポートの生成"""
        return {
            'privacy_budget': {
                'initial': self.config.epsilon,
                'spent': self.privacy_accountant['spent_budget'],
                'remaining': self.privacy_accountant['remaining_budget']
            },
            'noise_settings': {
                'multiplier': self.config.noise_multiplier,
                'max_grad_norm': self.config.max_grad_norm
            },
            'security_settings': {
                'encryption_method': self.config.encryption_method,
                'secure_aggregation': self.config.secure_aggregation
            },
            'timestamp': datetime.now().isoformat()
        }
