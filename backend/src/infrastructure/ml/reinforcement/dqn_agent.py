import random
from collections import deque
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self):
        # モデルのパラメータ
        self.state_size = 10  # 感情状態のベクトルサイズ
        self.action_size = 5  # 行動の種類数

        # DQNのハイパーパラメータ
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 割引率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        # デバイスの設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # モデルの初期化
        self.model = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = DQNNetwork(self.state_size, self.action_size).to(
            self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 行動の定義
        self.actions = [
            {"type": "challenge", "description": "より難しい問題を提供"},
            {"type": "support", "description": "ヒントやサポートを提供"},
            {"type": "encourage", "description": "励ましや動機付けを行う"},
            {"type": "break", "description": "休憩を提案"},
            {"type": "review", "description": "これまでの学習を振り返る"},
        ]

        # 感情の重み付け
        self.emotion_weights = {
            "joy": 1.0,
            "frustration": -0.5,
            "concentration": 0.8,
            "neutral": 0.0,
        }

    def _encode_state(self, state: Dict[str, Any]) -> np.ndarray:
        """状態を数値ベクトルに変換"""
        # 感情の種類をone-hotエンコーディング
        emotion_vector = np.zeros(7)  # 感情の種類数
        emotion_idx = list(self.emotion_weights.keys()).index(state["emotion"])
        emotion_vector[emotion_idx] = 1

        # 感情の強度を追加
        intensity = state["intensity"]

        # コンテキスト情報をエンコード
        context_vector = np.zeros(2)  # コンテキストの種類数
        if state["context"] == "learning":
            context_vector[0] = 1
        elif state["context"] == "challenge":
            context_vector[1] = 1

        # 全ての特徴を結合
        return np.concatenate([emotion_vector, [intensity], context_vector])

    def _calculate_reward(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        action: Dict[str, str],
    ) -> float:
        """報酬を計算"""
        # 感情の変化に基づく報酬
        old_emotion_value = self.emotion_weights.get(old_state["emotion"], 0)
        new_emotion_value = self.emotion_weights.get(new_state["emotion"], 0)
        emotion_change = new_emotion_value - old_emotion_value

        # 行動に基づく報酬
        action_rewards = {
            "challenge": (
                0.5 if new_state["emotion"] in ["joy", "concentration"] else -0.3
            ),
            "support": 0.5 if old_state["emotion"] == "frustration" else 0.0,
            "encourage": 0.3,
            "break": 0.2 if old_state["emotion"] == "frustration" else -0.1,
            "review": 0.4 if new_state["emotion"] in ["concentration", "joy"] else 0.1,
        }

        action_reward = action_rewards.get(action["type"], 0)

        # 総合報酬
        return emotion_change + action_reward

    def select_action(self, state: Dict[str, Any]) -> Dict[str, str]:
        """状態に基づいて行動を選択"""
        if random.random() < self.epsilon:
            # ランダムな行動を選択（探索）
            return random.choice(self.actions)

        # 状態をエンコード
        state_tensor = (
            torch.FloatTensor(self._encode_state(state)).unsqueeze(0).to(self.device)
        )

        # モデルで行動価値を予測
        with torch.no_grad():
            action_values = self.model(state_tensor)

        # 最適な行動を選択（活用）
        action_idx = torch.argmax(action_values).item()
        return self.actions[action_idx]

    def train(self, batch: List[Dict[str, Any]]) -> float:
        """バッチデータでモデルを学習"""
        if len(batch) < self.batch_size:
            return 0.0

        # バッチからランダムにサンプリング
        minibatch = random.sample(batch, self.batch_size)

        # データの準備
        states = torch.FloatTensor(
            [self._encode_state(d["state"]) for d in minibatch]
        ).to(self.device)
        actions = torch.LongTensor(
            [self.actions.index(d["action"]) for d in minibatch]
        ).to(self.device)
        rewards = torch.FloatTensor([d["reward"] for d in minibatch]).to(self.device)
        next_states = torch.FloatTensor(
            [self._encode_state(d["next_state"]) for d in minibatch]
        ).to(self.device)
        dones = torch.FloatTensor([d["done"] for d in minibatch]).to(self.device)

        # Q値の計算
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 損失の計算と最適化
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # εの減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_model(self):
        """ターゲットモデルを更新"""
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path: str):
        """モデルを保存"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load_model(self, path: str):
        """モデルを読み込み"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
