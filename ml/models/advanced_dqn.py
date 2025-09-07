import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import random
from collections import namedtuple, deque

# 経験リプレイのための型定義
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DuelingDQN(nn.Module):
    """Dueling DQNネットワーク"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DuelingDQN, self).__init__()
        
        # 特徴抽出層
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 価値関数（V）を推定する層
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # アドバンテージ関数（A）を推定する層
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """順伝播"""
        features = self.feature_layer(state)
        
        # 価値関数とアドバンテージ関数の計算
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q値の計算: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        qvalues = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvalues

class PrioritizedReplayBuffer:
    """優先度付き経験リプレイバッファ"""
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # 優先度の重み付け係数
        self.beta = beta    # 重要度サンプリングの係数
        self.beta_increment = 0.001  # βの増加量
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """経験の追加"""
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # 新しい経験には最大の優先度を与える
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """優先度に基づくサンプリング"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # 優先度に基づく確率の計算
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        
        # インデックスのサンプリング
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 重要度の計算
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # βの更新
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        experiences = [self.buffer[idx] for idx in indices]
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """優先度の更新"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self) -> int:
        return len(self.buffer)

class AdvancedDQNAgent:
    """拡張DQNエージェント"""
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 update_every: int = 4):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ネットワークの初期化
        self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # 優先度付き経験リプレイバッファの初期化
        self.memory = PrioritizedReplayBuffer(buffer_size)
        
        # パラメータの設定
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_every = update_every
        self.t_step = 0
        
        self.action_dim = action_dim
    
    def select_action(self, state: np.ndarray) -> Dict[str, int]:
        """行動の選択"""
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_indices = q_values.max(1)[1].cpu().numpy()
                
                # 行動の変換
                task_type = action_indices[0] // (5 * 3)
                remaining = action_indices[0] % (5 * 3)
                difficulty = remaining // 3
                feedback_style = remaining % 3
                
                return {
                    'task_type': int(task_type),
                    'difficulty': int(difficulty),
                    'feedback_style': int(feedback_style)
                }
        else:
            # ランダムな行動の生成
            return {
                'task_type': random.randint(0, 3),
                'difficulty': random.randint(0, 4),
                'feedback_style': random.randint(0, 2)
            }
    
    def _action_to_index(self, action: Dict[str, int]) -> int:
        """行動辞書からインデックスへの変換"""
        return (action['task_type'] * 5 * 3 +
                action['difficulty'] * 3 +
                action['feedback_style'])
    
    def _process_experience(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        """状態辞書から状態ベクトルへの変換"""
        return np.concatenate([
            state['perseverance'],
            state['self_control'],
            state['cooperation'],
            state['curiosity'],
            [state['current_level']],
            [state['challenge_count']],
            state['success_rate'],
            state['emotional_state']
        ])
    
    def step(self, state: Dict[str, np.ndarray], action: Dict[str, int],
             reward: float, next_state: Dict[str, np.ndarray], done: bool):
        """学習ステップの実行"""
        # 状態の前処理
        state_vector = self._process_experience(state)
        next_state_vector = self._process_experience(next_state)
        action_index = self._action_to_index(action)
        
        # 経験の保存
        self.memory.push(state_vector, action_index, reward, next_state_vector, done)
        
        # 定期的な学習の実行
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self._learn()
        
        # εの更新
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _learn(self):
        """バッチ学習の実行"""
        experiences, indices, weights = self.memory.sample(self.batch_size)
        if len(experiences) == 0:
            return
        
        # バッチデータの準備
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Q値の計算
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).detach().max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # TD誤差の計算
        td_errors = torch.abs(current_q_values - target_q_values.unsqueeze(1)).cpu().data.numpy()
        
        # 優先度の更新
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # 損失の計算と最適化
        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(
            current_q_values, target_q_values.unsqueeze(1), reduction='none'
        )).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # ターゲットネットワークの更新
        self._soft_update(self.policy_net, self.target_net)
    
    def _soft_update(self, local_model: nn.Module, target_model: nn.Module):
        """ターゲットネットワークのソフトアップデート"""
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """モデルの保存"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """モデルの読み込み"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
