import pytest
import numpy as np
from ..models.learning_environment import NonCogLearningEnvironment

@pytest.fixture
def env():
    return NonCogLearningEnvironment()

def test_environment_initialization(env):
    """環境の初期化テスト"""
    assert isinstance(env.state, dict)
    assert all(key in env.state for key in [
        'perseverance', 'self_control', 'cooperation', 'curiosity',
        'current_level', 'challenge_count', 'success_rate', 'emotional_state'
    ])
    
    # 初期値の確認
    assert env.state['perseverance'][0] == 0.5
    assert env.state['current_level'] == 0
    assert env.state['emotional_state'][0] == 0.0

def test_action_space(env):
    """行動空間のテスト"""
    assert hasattr(env.action_space, 'spaces')
    assert 'task_type' in env.action_space.spaces
    assert 'difficulty' in env.action_space.spaces
    assert 'feedback_style' in env.action_space.spaces
    
    # 行動空間の範囲確認
    assert env.action_space.spaces['task_type'].n == 4
    assert env.action_space.spaces['difficulty'].n == 5
    assert env.action_space.spaces['feedback_style'].n == 3

def test_step_execution(env):
    """ステップ実行のテスト"""
    action = {
        'task_type': 0,
        'difficulty': 1,
        'feedback_style': 0
    }
    
    state, reward, done, info = env.step(action)
    
    # 状態の型と範囲の確認
    assert isinstance(state, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    # 状態値の範囲確認
    assert 0 <= state['perseverance'][0] <= 1
    assert 0 <= state['emotional_state'][0] <= 1
    assert state['challenge_count'] > 0

def test_reward_calculation(env):
    """報酬計算のテスト"""
    action = {
        'task_type': 0,
        'difficulty': 4,  # 高難度
        'feedback_style': 0
    }
    
    # 複数回ステップを実行して報酬の範囲を確認
    rewards = []
    for _ in range(10):
        _, reward, _, _ = env.step(action)
        rewards.append(reward)
    
    # 報酬の統計的な確認
    assert min(rewards) >= -1.0  # 最小報酬の確認
    assert max(rewards) <= 3.0   # 最大報酬の確認

def test_episode_termination(env):
    """エピソード終了条件のテスト"""
    action = {
        'task_type': 0,
        'difficulty': 1,
        'feedback_style': 0
    }
    
    # max_episode_stepsまでステップを実行
    done = False
    steps = 0
    while not done and steps < env.max_episode_steps + 10:
        _, _, done, _ = env.step(action)
        steps += 1
    
    assert steps == env.max_episode_steps
    assert done

def test_state_updates(env):
    """状態更新の一貫性テスト"""
    initial_state = env.reset()
    
    action = {
        'task_type': 0,
        'difficulty': 1,
        'feedback_style': 0
    }
    
    next_state, _, _, _ = env.step(action)
    
    # チャレンジカウントの更新確認
    assert next_state['challenge_count'] == initial_state['challenge_count'] + 1
    
    # 非認知能力値の範囲確認
    for ability in ['perseverance', 'self_control', 'cooperation', 'curiosity']:
        assert 0 <= next_state[ability][0] <= 1

def test_difficulty_impact(env):
    """難易度の影響テスト"""
    env.reset()
    
    # 低難度アクション
    easy_action = {
        'task_type': 0,
        'difficulty': 1,
        'feedback_style': 0
    }
    
    # 高難度アクション
    hard_action = {
        'task_type': 0,
        'difficulty': 4,
        'feedback_style': 0
    }
    
    # 各難易度で複数回試行
    easy_successes = []
    hard_successes = []
    
    for _ in range(20):
        env.reset()
        _, _, _, info_easy = env.step(easy_action)
        easy_successes.append(info_easy['task_outcome']['success'])
        
        env.reset()
        _, _, _, info_hard = env.step(hard_action)
        hard_successes.append(info_hard['task_outcome']['success'])
    
    # 低難度の方が成功率が高いことを確認
    assert np.mean(easy_successes) > np.mean(hard_successes)

def test_emotional_state_changes(env):
    """感情状態の変化テスト"""
    env.reset()
    initial_emotion = env.state['emotional_state'][0]
    
    action = {
        'task_type': 0,
        'difficulty': 2,
        'feedback_style': 0
    }
    
    # 複数回ステップを実行して感情状態の変化を確認
    emotions = [initial_emotion]
    for _ in range(5):
        next_state, _, _, _ = env.step(action)
        emotions.append(next_state['emotional_state'][0])
    
    # 感情状態が変化していることを確認
    assert len(set(emotions)) > 1
    # 感情状態が範囲内に収まっていることを確認
    assert all(-1 <= e <= 1 for e in emotions)

def test_reset_functionality(env):
    """リセット機能のテスト"""
    initial_state = env.reset()
    
    # いくつかのステップを実行
    action = {
        'task_type': 0,
        'difficulty': 1,
        'feedback_style': 0
    }
    
    for _ in range(5):
        env.step(action)
    
    # リセット後の状態を確認
    reset_state = env.reset()
    
    # 全ての状態値が初期値に戻っていることを確認
    for key in initial_state:
        np.testing.assert_array_equal(initial_state[key], reset_state[key])
