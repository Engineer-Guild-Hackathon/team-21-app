import numpy as np
from typing import Tuple, Dict, Any
import gym
from gym import spaces
import json
import logging

class NonCogLearningEnvironment(gym.Env):
    """非認知能力学習のための強化学習環境"""
    
    def __init__(self):
        super(NonCogLearningEnvironment, self).__init__()
        
        # 状態空間の定義
        self.observation_space = spaces.Dict({
            # 非認知能力スコア（0-1の範囲）
            'perseverance': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # やり抜く力
            'self_control': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # 自制心
            'cooperation': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),   # 協調性
            'curiosity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),     # 好奇心
            
            # 学習進捗
            'current_level': spaces.Discrete(10),  # 現在のレベル
            'challenge_count': spaces.Discrete(100),  # チャレンジ回数
            'success_rate': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # 成功率
            
            # 感情状態（-1: ネガティブ, 0: 中立, 1: ポジティブ）
            'emotional_state': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        })
        
        # 行動空間の定義
        self.action_space = spaces.Dict({
            # タスクの種類
            'task_type': spaces.Discrete(4),  # [問題解決, 協力課題, 創造的課題, 自己管理課題]
            
            # タスクの難易度
            'difficulty': spaces.Discrete(5),  # [1-5の難易度]
            
            # フィードバックスタイル
            'feedback_style': spaces.Discrete(3),  # [励まし重視, 解説重視, バランス型]
        })
        
        # 環境の初期状態
        self.state = self._get_initial_state()
        self.episode_steps = 0
        self.max_episode_steps = 100
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _get_initial_state(self) -> Dict[str, np.ndarray]:
        """初期状態の生成"""
        return {
            'perseverance': np.array([0.5], dtype=np.float32),
            'self_control': np.array([0.5], dtype=np.float32),
            'cooperation': np.array([0.5], dtype=np.float32),
            'curiosity': np.array([0.5], dtype=np.float32),
            'current_level': 0,
            'challenge_count': 0,
            'success_rate': np.array([0.5], dtype=np.float32),
            'emotional_state': np.array([0.0], dtype=np.float32),
        }
    
    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """環境の1ステップ実行"""
        self.episode_steps += 1
        
        # アクションの実行と結果の計算
        task_outcome = self._execute_task(action)
        
        # 状態の更新
        self._update_state(action, task_outcome)
        
        # 報酬の計算
        reward = self._calculate_reward(action, task_outcome)
        
        # エピソード終了判定
        done = self.episode_steps >= self.max_episode_steps
        
        # 追加情報
        info = {
            'task_outcome': task_outcome,
            'episode_steps': self.episode_steps,
        }
        
        return self.state, reward, done, info
    
    def _execute_task(self, action: Dict[str, int]) -> Dict[str, float]:
        """タスクの実行とその結果の計算"""
        # タスクの成功確率を計算
        base_success_prob = self._calculate_base_success_probability(action)
        
        # 非認知能力による補正
        ability_modifier = self._calculate_ability_modifier()
        
        # 最終的な成功確率
        final_success_prob = np.clip(base_success_prob * ability_modifier, 0.1, 0.9)
        
        # タスクの結果を決定
        success = np.random.random() < final_success_prob
        
        # 感情状態の変化を計算
        emotional_change = self._calculate_emotional_change(success, action)
        
        return {
            'success': success,
            'emotional_change': emotional_change,
            'ability_gain': self._calculate_ability_gain(success, action),
        }
    
    def _calculate_base_success_probability(self, action: Dict[str, int]) -> float:
        """基本成功確率の計算"""
        difficulty_factor = 1.0 - (action['difficulty'] / 5.0)
        level_factor = min(1.0, (self.state['current_level'] + 1) / 10.0)
        return 0.7 * difficulty_factor + 0.3 * level_factor
    
    def _calculate_ability_modifier(self) -> float:
        """非認知能力による成功確率の補正値を計算"""
        abilities = [
            self.state['perseverance'][0],
            self.state['self_control'][0],
            self.state['cooperation'][0],
            self.state['curiosity'][0]
        ]
        return np.mean(abilities)
    
    def _calculate_emotional_change(self, success: bool, action: Dict[str, int]) -> float:
        """感情状態の変化量を計算"""
        base_change = 0.1 if success else -0.05
        difficulty_bonus = action['difficulty'] / 10.0 if success else 0
        return base_change + difficulty_bonus
    
    def _calculate_ability_gain(self, success: bool, action: Dict[str, int]) -> Dict[str, float]:
        """非認知能力の向上量を計算"""
        base_gain = 0.02 if success else 0.01
        difficulty_multiplier = 1.0 + (action['difficulty'] / 10.0)
        
        gains = {
            'perseverance': base_gain * difficulty_multiplier,
            'self_control': base_gain * difficulty_multiplier,
            'cooperation': base_gain * difficulty_multiplier,
            'curiosity': base_gain * difficulty_multiplier
        }
        
        # タスクタイプに応じて特定の能力の向上を強化
        task_type = action['task_type']
        if task_type == 0:  # 問題解決
            gains['perseverance'] *= 1.5
        elif task_type == 1:  # 協力課題
            gains['cooperation'] *= 1.5
        elif task_type == 2:  # 創造的課題
            gains['curiosity'] *= 1.5
        elif task_type == 3:  # 自己管理課題
            gains['self_control'] *= 1.5
            
        return gains
    
    def _update_state(self, action: Dict[str, int], task_outcome: Dict[str, Any]):
        """状態の更新"""
        # 非認知能力の更新
        ability_gains = task_outcome['ability_gain']
        for ability, gain in ability_gains.items():
            current_value = self.state[ability][0]
            self.state[ability] = np.array([min(1.0, current_value + gain)], dtype=np.float32)
        
        # 感情状態の更新
        current_emotion = self.state['emotional_state'][0]
        new_emotion = np.clip(current_emotion + task_outcome['emotional_change'], -1, 1)
        self.state['emotional_state'] = np.array([new_emotion], dtype=np.float32)
        
        # 進捗の更新
        self.state['challenge_count'] += 1
        if task_outcome['success']:
            self.state['current_level'] = min(9, self.state['current_level'] + 1)
        
        # 成功率の更新
        current_success_rate = self.state['success_rate'][0]
        new_success_rate = (current_success_rate * (self.state['challenge_count'] - 1) + 
                          float(task_outcome['success'])) / self.state['challenge_count']
        self.state['success_rate'] = np.array([new_success_rate], dtype=np.float32)
    
    def _calculate_reward(self, action: Dict[str, int], task_outcome: Dict[str, Any]) -> float:
        """報酬の計算"""
        # 基本報酬（タスクの成功/失敗）
        base_reward = 1.0 if task_outcome['success'] else -0.1
        
        # 難易度ボーナス
        difficulty_bonus = action['difficulty'] / 5.0 if task_outcome['success'] else 0
        
        # 感情状態ボーナス（ポジティブな感情を維持することを奨励）
        emotion_bonus = max(0, self.state['emotional_state'][0]) * 0.2
        
        # 非認知能力の向上ボーナス
        ability_bonus = sum(task_outcome['ability_gain'].values()) * 2.0
        
        total_reward = base_reward + difficulty_bonus + emotion_bonus + ability_bonus
        return float(total_reward)
    
    def reset(self) -> Dict[str, np.ndarray]:
        """環境のリセット"""
        self.state = self._get_initial_state()
        self.episode_steps = 0
        return self.state
    
    def render(self, mode='human'):
        """環境の状態を表示"""
        if mode == 'human':
            state_info = {
                'Perseverance': float(self.state['perseverance'][0]),
                'Self Control': float(self.state['self_control'][0]),
                'Cooperation': float(self.state['cooperation'][0]),
                'Curiosity': float(self.state['curiosity'][0]),
                'Current Level': self.state['current_level'],
                'Challenge Count': self.state['challenge_count'],
                'Success Rate': float(self.state['success_rate'][0]),
                'Emotional State': float(self.state['emotional_state'][0])
            }
            print(json.dumps(state_info, indent=2))
