"""
DQN (Deep Q-Network) エージェント
プレイヤーの感情状態と学習進捗に基づいて最適なアクションを選択する
"""
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        """
        Q-Networkの構造を定義
        
        Args:
            state_size: 状態の次元数（感情状態 + 学習進捗などの特徴量）
            action_size: 行動の種類数（提供可能なクエストやヒントの数）
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        順伝播の実装
        
        Args:
            state: 入力状態
            
        Returns:
            各行動に対するQ値
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        """
        DQNエージェントの初期化
        
        Args:
            state_size: 状態の次元数
            action_size: 行動の種類数
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-Network（メインネットワークとターゲットネットワーク）
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters())
        
        # リプレイメモリ
        self.memory = deque(maxlen=10000)
        
        # 学習パラメータ
        self.batch_size = 64
        self.gamma = 0.99    # 割引率
        self.tau = 1e-3     # ソフトアップデート係数
        self.epsilon = 1.0   # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        現在の状態に基づいて行動を選択
        
        Args:
            state: 現在の状態
            eval_mode: 評価モードの場合True（グリーディに行動選択）
            
        Returns:
            選択された行動のインデックス
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        return np.argmax(action_values.cpu().data.numpy())
    
    def step(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """
        1ステップの経験を保存し、学習を実行
        
        Args:
            state: 現在の状態
            action: 実行した行動
            reward: 得られた報酬
            next_state: 遷移後の状態
            done: エピソード終了フラグ
        """
        # 経験をメモリに保存
        self.memory.append((state, action, reward, next_state, done))
        
        # メモリが十分たまっていれば学習を実行
        if len(self.memory) > self.batch_size:
            self._learn()
            
    def _learn(self):
        """バッチ学習を実行"""
        # メモリからランダムにバッチをサンプリング
        experiences = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
        
        # Q値の計算
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # 損失計算と最適化
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ターゲットネットワークのソフトアップデート
        self._soft_update()
        
        # 探索率の減衰
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def _soft_update(self):
        """ターゲットネットワークのソフトアップデート"""
        for target_param, local_param in zip(self.qnetwork_target.parameters(),
                                           self.qnetwork_local.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
    def get_action_for_emotional_state(self, 
                                     emotional_state: Dict[str, float],
                                     learning_progress: Dict[str, float]) -> Dict:
        """
        感情状態と学習進捗に基づいて最適なアクションを選択
        
        Args:
            emotional_state: 感情状態の辞書
            learning_progress: 学習進捗の辞書
            
        Returns:
            選択されたアクションの詳細
        """
        # 状態の特徴量化
        state = np.array([
            emotional_state.get('joy', 0.0),
            emotional_state.get('sadness', 0.0),
            emotional_state.get('anger', 0.0),
            emotional_state.get('fear', 0.0),
            emotional_state.get('surprise', 0.0),
            emotional_state.get('frustration', 0.0),
            emotional_state.get('concentration', 0.0),
            learning_progress.get('current_level', 0.0),
            learning_progress.get('success_rate', 0.0),
            learning_progress.get('attempt_count', 0.0)
        ])
        
        # 行動の選択
        action_idx = self.act(state, eval_mode=True)
        
        # 行動の種類（例）
        actions = [
            {
                'type': 'easy_quest',
                'description': '気分転換のための簡単なクエスト提供'
            },
            {
                'type': 'hint',
                'description': 'ヒントを提供するNPCとの会話'
            },
            {
                'type': 'challenge',
                'description': '難易度の高い問題の提供'
            },
            {
                'type': 'skill_quest',
                'description': '新しいスキル習得のためのクエスト'
            },
            {
                'type': 'break',
                'description': '小休憩の提案'
            }
        ]
        
        return actions[action_idx]
