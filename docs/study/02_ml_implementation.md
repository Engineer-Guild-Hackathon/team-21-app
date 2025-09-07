# 機械学習実装ガイド

## 強化学習（DQN）

### 1. モデル定義

```python
# ml/models/dqn_model.py
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )

        # 価値関数
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # アドバンテージ関数
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean())
```

### 2. 経験リプレイ

```python
# ml/models/replay_buffer.py
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities[len(self.buffer) - 1] = max(self.priorities)

    def sample(self, batch_size: int):
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        return map(list, zip(*samples)), indices
```

### 3. エージェント実装

```python
# ml/models/dqn_agent.py
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.policy_net = DuelingDQN(state_dim, action_dim)
        self.target_net = DuelingDQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = PrioritizedReplayBuffer(10000)

    def select_action(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def update(self, batch_size: int):
        (states, actions, rewards, next_states, dones), indices = \
            self.memory.sample(batch_size)

        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach()
        expected_q = rewards + 0.99 * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, expected_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

## 感情分析（BERT）

### 1. モデル定義

```python
# ml/models/emotion_model.py
from transformers import BertModel

class EmotionClassifier(nn.Module):
    def __init__(self, num_emotions: int):
        super().__init__()
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
        self.classifier = nn.Linear(768, num_emotions)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.classifier(outputs.pooler_output)
```

### 2. データ処理

```python
# ml/data/emotion_dataset.py
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx]
            for key, val in self.encodings.items()
        }
        item['labels'] = self.labels[idx]
        return item
```

### 3. 学習ループ

```python
# ml/training/emotion_trainer.py
def train_emotion_model(model, train_loader, val_loader, num_epochs):
    optimizer = optim.AdamW(model.parameters())
    scheduler = get_linear_schedule_with_warmup(optimizer, num_epochs)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            loss = F.cross_entropy(outputs, batch['labels'])
            loss.backward()
            optimizer.step()

        scheduler.step()
```

## フェデレーテッドラーニング

### 1. クライアント実装

```python
# ml/federated/client.py
class FederatedClient:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)

    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            for batch in data_loader:
                self.optimizer.zero_grad()
                loss = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()

    def get_model_update(self):
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
```

### 2. サーバー実装

```python
# ml/federated/server.py
class FederatedServer:
    def __init__(self, model):
        self.global_model = model

    def aggregate_updates(self, client_updates):
        averaged_update = {}
        num_clients = len(client_updates)

        for name, param in self.global_model.named_parameters():
            averaged_update[name] = torch.stack([
                updates[name] for updates in client_updates
            ]).mean(0)

        return averaged_update

    def update_global_model(self, averaged_update):
        for name, param in self.global_model.named_parameters():
            param.data.copy_(averaged_update[name])
```

### 3. プライバシー保護

```python
# ml/federated/privacy.py
class DifferentialPrivacy:
    def __init__(self, epsilon: float, delta: float):
        self.epsilon = epsilon
        self.delta = delta

    def add_noise(self, parameters):
        sensitivity = self.compute_sensitivity(parameters)
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

        noised_parameters = {}
        for name, param in parameters.items():
            noise = torch.randn_like(param) * noise_scale
            noised_parameters[name] = param + noise

        return noised_parameters
```

## MLOps 実装

### 1. モデル管理

```python
# ml/mlops/model_registry.py
class ModelRegistry:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)

    def save_model(self, model, metadata):
        version = self._get_next_version()
        save_path = self.storage_path / f"model_v{version}"

        torch.save({
            'state_dict': model.state_dict(),
            'metadata': metadata
        }, save_path)

    def load_model(self, version: str):
        load_path = self.storage_path / f"model_v{version}"
        checkpoint = torch.load(load_path)
        return checkpoint['state_dict'], checkpoint['metadata']
```

### 2. 実験管理

```python
# ml/mlops/experiment_tracker.py
import mlflow

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def log_metrics(self, metrics: Dict[str, float]):
        mlflow.log_metrics(metrics)

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_model(self, model, artifact_path: str):
        mlflow.pytorch.log_model(model, artifact_path)
```

### 3. モニタリング

```python
# ml/mlops/monitoring.py
from prometheus_client import Counter, Histogram

class ModelMonitoring:
    def __init__(self):
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Prediction latency in seconds'
        )

        self.prediction_errors = Counter(
            'model_prediction_errors_total',
            'Total prediction errors'
        )

    @contextmanager
    def measure_latency(self):
        start_time = time.time()
        try:
            yield
        finally:
            latency = time.time() - start_time
            self.prediction_latency.observe(latency)

    def record_error(self):
        self.prediction_errors.inc()
```
