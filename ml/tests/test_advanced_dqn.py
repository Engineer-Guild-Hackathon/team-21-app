import numpy as np
import pytest
import torch

from ..models.advanced_dqn import (AdvancedDQNAgent, DuelingDQN,
                                   PrioritizedReplayBuffer)
from ..models.learning_environment import NonCogLearningEnvironment


@pytest.fixture
def env():
    return NonCogLearningEnvironment()


@pytest.fixture
def agent(env):
    state_dim = (
        8  # 非認知能力(4) + レベル(1) + チャレンジ数(1) + 成功率(1) + 感情状態(1)
    )
    action_dim = 4 * 5 * 3  # task_type(4) × difficulty(5) × feedback_style(3)
    return AdvancedDQNAgent(state_dim, action_dim)


def test_dueling_dqn_architecture():
    """Dueling DQNアーキテクチャのテスト"""
    state_dim = 8
    action_dim = 60
    model = DuelingDQN(state_dim, action_dim)

    # 入力テンソルの作成
    batch_size = 32
    state = torch.randn(batch_size, state_dim)

    # 順伝播
    q_values = model(state)

    # 出力の形状確認
    assert q_values.shape == (batch_size, action_dim)
    assert torch.isfinite(q_values).all()  # 出力値が有限であることを確認


def test_prioritized_replay_buffer():
    """優先度付き経験リプレイバッファのテスト"""
    buffer = PrioritizedReplayBuffer(capacity=100)

    # データの追加
    state = np.random.randn(8)
    action = 0
    reward = 1.0
    next_state = np.random.randn(8)
    done = False

    # バッファに経験を追加
    for _ in range(50):
        buffer.push(state, action, reward, next_state, done)

    # サンプリングのテスト
    batch_size = 32
    experiences, indices, weights = buffer.sample(batch_size)

    assert len(experiences) == batch_size
    assert len(indices) == batch_size
    assert len(weights) == batch_size
    assert np.all(weights > 0)  # 重みが正であることを確認


def test_agent_action_selection(agent, env):
    """エージェントの行動選択のテスト"""
    state = env.reset()
    state_vector = agent._process_experience(state)

    # 決定論的な行動選択（ε = 0）
    agent.epsilon = 0
    action = agent.select_action(state_vector)

    assert isinstance(action, dict)
    assert 0 <= action["task_type"] <= 3
    assert 0 <= action["difficulty"] <= 4
    assert 0 <= action["feedback_style"] <= 2

    # ランダムな行動選択（ε = 1）
    agent.epsilon = 1
    action = agent.select_action(state_vector)

    assert isinstance(action, dict)
    assert 0 <= action["task_type"] <= 3
    assert 0 <= action["difficulty"] <= 4
    assert 0 <= action["feedback_style"] <= 2


def test_agent_learning(agent, env):
    """エージェントの学習プロセスのテスト"""
    # 経験の収集
    state = env.reset()
    state_vector = agent._process_experience(state)

    for _ in range(100):
        action = agent.select_action(state_vector)
        next_state, reward, done, _ = env.step(action)
        next_state_vector = agent._process_experience(next_state)

        # 学習ステップの実行
        agent.step(state, action, reward, next_state, done)

        if done:
            state = env.reset()
            state_vector = agent._process_experience(state)
        else:
            state = next_state
            state_vector = next_state_vector

    # バッファに十分なデータが蓄積されていることを確認
    assert len(agent.memory) >= agent.batch_size


def test_model_save_load(agent, tmp_path):
    """モデルの保存と読み込みのテスト"""
    # モデルの保存
    save_path = tmp_path / "model.pth"
    agent.save(str(save_path))

    # 新しいエージェントの作成
    new_agent = AdvancedDQNAgent(8, 60)

    # モデルの読み込み
    new_agent.load(str(save_path))

    # パラメータが正しく読み込まれたことを確認
    for p1, p2 in zip(agent.policy_net.parameters(), new_agent.policy_net.parameters()):
        assert torch.equal(p1, p2)

    assert agent.epsilon == new_agent.epsilon


def test_action_conversion(agent):
    """行動の変換テスト"""
    # 辞書からインデックスへの変換
    action_dict = {"task_type": 2, "difficulty": 3, "feedback_style": 1}

    index = agent._action_to_index(action_dict)

    # インデックスが期待される範囲内にあることを確認
    assert 0 <= index < 60  # 4 * 5 * 3 = 60


def test_state_processing(agent, env):
    """状態の処理テスト"""
    state = env.reset()
    state_vector = agent._process_experience(state)

    # 状態ベクトルの形状確認
    assert isinstance(state_vector, np.ndarray)
    assert state_vector.shape == (8,)  # 期待される状態の次元
    assert np.all(np.isfinite(state_vector))  # 全ての値が有限であることを確認


def test_soft_update(agent):
    """ソフトアップデートのテスト"""
    # ポリシーネットワークのパラメータを変更
    for param in agent.policy_net.parameters():
        param.data.fill_(1.0)

    # ターゲットネットワークのパラメータを初期化
    for param in agent.target_net.parameters():
        param.data.fill_(0.0)

    # ソフトアップデートの実行
    agent._soft_update(agent.policy_net, agent.target_net)

    # パラメータが期待通りに更新されていることを確認
    for target_param in agent.target_net.parameters():
        assert torch.allclose(target_param, torch.tensor(agent.tau))


def test_epsilon_decay(agent):
    """εの減衰テスト"""
    initial_epsilon = agent.epsilon

    # 複数回のステップを実行
    state = np.random.randn(8)
    action = {"task_type": 0, "difficulty": 0, "feedback_style": 0}
    next_state = np.random.randn(8)

    for _ in range(100):
        agent.step(state, action, 0.0, next_state, False)

    # εが減衰していることを確認
    assert agent.epsilon < initial_epsilon
    assert agent.epsilon >= agent.epsilon_end
