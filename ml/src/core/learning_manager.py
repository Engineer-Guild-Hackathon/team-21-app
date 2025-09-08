"""
学習管理システム
感情分析、強化学習、対話システムを統合して学習体験を管理する
"""
from typing import Dict, List, Optional
import json
from ..emotion_analysis.emotion_analyzer import EmotionAnalyzer
from ..reinforcement.dqn_agent import DQNAgent
from ..dialogue.bert_dialogue import DialogueSystem

class LearningManager:
    def __init__(self):
        """学習管理システムの初期化"""
        # 各コンポーネントの初期化
        self.emotion_analyzer = EmotionAnalyzer()
        self.dqn_agent = DQNAgent(state_size=10, action_size=5)
        self.dialogue_system = DialogueSystem()
        
        # 学習状態の初期化
        self.learning_state = {
            'current_level': 1,
            'success_rate': 0.0,
            'attempt_count': 0,
            'completed_quests': [],
            'skills_acquired': set(),
            'dialogue_history': []
        }
        
    def process_user_interaction(self,
                               text: Optional[str] = None,
                               facial_expression: Optional[Dict] = None,
                               behavior_data: Optional[Dict] = None) -> Dict:
        """
        ユーザーの入力を処理し、適切なレスポンスを生成
        
        Args:
            text: ユーザーの入力テキスト
            facial_expression: 表情分析データ
            behavior_data: 行動データ
            
        Returns:
            Dict: レスポンス情報
        """
        # 感情状態の分析
        emotional_state = self.emotion_analyzer.get_emotional_state(
            text=text,
            facial_expression=facial_expression,
            behavior_data=behavior_data
        )
        
        # 強化学習による次のアクションの決定
        next_action = self.dqn_agent.get_action_for_emotional_state(
            emotional_state=emotional_state,
            learning_progress=self.learning_state
        )
        
        # 対話システムによる応答生成
        if text:
            self.learning_state['dialogue_history'].append(text)
            dialogue_response = self.dialogue_system.generate_response(
                user_input=text,
                emotional_state=emotional_state,
                dialogue_history=self.learning_state['dialogue_history']
            )
        else:
            dialogue_response = None
            
        # 非認知能力の評価
        if len(self.learning_state['dialogue_history']) > 0:
            noncog_scores = self.dialogue_system.evaluate_noncognitive_skills(
                self.learning_state['dialogue_history']
            )
        else:
            noncog_scores = {}
            
        return {
            'emotional_state': emotional_state,
            'next_action': next_action,
            'dialogue_response': dialogue_response,
            'noncog_evaluation': noncog_scores,
            'feedback': self.emotion_analyzer.get_feedback_suggestion(emotional_state)
        }
        
    def update_learning_progress(self,
                               quest_result: Dict[str, any],
                               reward: float):
        """
        学習進捗を更新
        
        Args:
            quest_result: クエスト結果の情報
            reward: 報酬値
        """
        # 学習状態の更新
        self.learning_state['attempt_count'] += 1
        if quest_result.get('success', False):
            self.learning_state['completed_quests'].append(quest_result['quest_id'])
            self.learning_state['success_rate'] = (
                len(self.learning_state['completed_quests']) /
                self.learning_state['attempt_count']
            )
            
        # 新しいスキルの追加
        if 'acquired_skills' in quest_result:
            self.learning_state['skills_acquired'].update(
                quest_result['acquired_skills']
            )
            
        # レベルの更新
        if self.learning_state['success_rate'] > 0.7 and \
           len(self.learning_state['completed_quests']) > 5:
            self.learning_state['current_level'] += 1
            
        # DQNエージェントの更新
        current_state = self._get_state_vector()
        next_state = self._get_state_vector()  # 更新後の状態
        
        self.dqn_agent.step(
            state=current_state,
            action=quest_result.get('action_taken', 0),
            reward=reward,
            next_state=next_state,
            done=quest_result.get('quest_completed', False)
        )
        
    def _get_state_vector(self) -> np.ndarray:
        """
        現在の学習状態をベクトル化
        
        Returns:
            np.ndarray: 状態ベクトル
        """
        return np.array([
            self.learning_state['current_level'],
            self.learning_state['success_rate'],
            self.learning_state['attempt_count'],
            len(self.learning_state['completed_quests']),
            len(self.learning_state['skills_acquired']),
            # 追加の特徴量があれば追加
        ])
