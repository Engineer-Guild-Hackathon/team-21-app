import pytest
from ..models.feedback_generator import (
    FeedbackGenerator,
    FeedbackContext,
    FeedbackStyle,
    FeedbackTemplate
)
from ..models.emotion_analyzer import EmotionCategory, EmotionAnalysisResult
from dataclasses import dataclass
import torch

@pytest.fixture
def feedback_generator():
    return FeedbackGenerator()

@pytest.fixture
def mock_emotion_result():
    @dataclass
    class MockEmotionResult:
        primary_emotion: EmotionCategory
        emotion_scores: dict
        confidence: float
    
    return MockEmotionResult(
        primary_emotion=EmotionCategory.MOTIVATED,
        emotion_scores={emotion: 0.1 for emotion in EmotionCategory},
        confidence=0.8
    )

@pytest.fixture
def mock_learning_state():
    return {
        'current_level': 3,
        'success_rate': 0.75,
        'challenge_count': 10,
        'focus_duration': 45,
        'help_requests': 2
    }

@pytest.fixture
def mock_task_history():
    return [
        {
            'task_id': 1,
            'description': '基本問題の解決',
            'success': True,
            'time_spent': 300
        },
        {
            'task_id': 2,
            'description': '応用問題への挑戦',
            'success': False,
            'time_spent': 450
        },
        {
            'task_id': 3,
            'description': '協力課題の完了',
            'success': True,
            'time_spent': 600
        }
    ]

def test_feedback_generator_initialization(feedback_generator):
    """フィードバック生成器の初期化テスト"""
    assert feedback_generator.model is not None
    assert feedback_generator.tokenizer is not None
    assert isinstance(feedback_generator.template, FeedbackTemplate)

def test_feedback_style_determination(feedback_generator, mock_emotion_result,
                                    mock_learning_state, mock_task_history):
    """フィードバックスタイルの決定テスト"""
    context = FeedbackContext(
        emotion_state=mock_emotion_result,
        learning_state=mock_learning_state,
        task_history=mock_task_history,
        preferred_style=None
    )
    
    style = feedback_generator._determine_feedback_style(context)
    assert isinstance(style, FeedbackStyle)

def test_template_selection(feedback_generator):
    """テンプレート選択のテスト"""
    template = feedback_generator._select_template(
        EmotionCategory.MOTIVATED,
        FeedbackStyle.ENCOURAGING
    )
    
    assert isinstance(template, str)
    assert len(template) > 0
    assert '{' in template and '}' in template  # プレースホルダーの存在確認

def test_context_info_extraction(feedback_generator, mock_emotion_result,
                               mock_learning_state, mock_task_history):
    """コンテキスト情報抽出のテスト"""
    context = FeedbackContext(
        emotion_state=mock_emotion_result,
        learning_state=mock_learning_state,
        task_history=mock_task_history,
        preferred_style=None
    )
    
    context_info = feedback_generator._extract_context_info(context)
    
    assert isinstance(context_info, dict)
    assert 'achievement' in context_info
    assert 'next_goal' in context_info
    assert 'progress' in context_info
    assert all(isinstance(value, str) for value in context_info.values())

def test_feedback_generation(feedback_generator, mock_emotion_result,
                           mock_learning_state, mock_task_history):
    """フィードバック生成の統合テスト"""
    context = FeedbackContext(
        emotion_state=mock_emotion_result,
        learning_state=mock_learning_state,
        task_history=mock_task_history,
        preferred_style=FeedbackStyle.ENCOURAGING
    )
    
    feedback = feedback_generator.generate_feedback(context)
    
    assert isinstance(feedback, str)
    assert len(feedback) > 0
    assert not any(char in feedback for char in ['{', '}'])  # プレースホルダーが全て置換されていることを確認

def test_recent_achievements_extraction(feedback_generator, mock_task_history):
    """最近の成果抽出のテスト"""
    achievements = feedback_generator._extract_recent_achievements(mock_task_history)
    
    assert isinstance(achievements, list)
    assert len(achievements) > 0
    assert all(isinstance(achievement, str) for achievement in achievements)
    assert any('基本問題' in achievement for achievement in achievements)

def test_next_goals_determination(feedback_generator, mock_emotion_result,
                                mock_learning_state, mock_task_history):
    """次の目標決定のテスト"""
    context = FeedbackContext(
        emotion_state=mock_emotion_result,
        learning_state=mock_learning_state,
        task_history=mock_task_history,
        preferred_style=None
    )
    
    goals = feedback_generator._determine_next_goals(context)
    
    assert isinstance(goals, list)
    assert len(goals) > 0
    assert all(isinstance(goal, str) for goal in goals)

def test_feedback_enhancement(feedback_generator, mock_emotion_result,
                            mock_learning_state, mock_task_history):
    """フィードバック強化のテスト"""
    context = FeedbackContext(
        emotion_state=mock_emotion_result,
        learning_state=mock_learning_state,
        task_history=mock_task_history,
        preferred_style=None
    )
    
    base_feedback = "頑張っていますね。次も頑張りましょう。"
    enhanced_feedback = feedback_generator._enhance_feedback(base_feedback, context)
    
    assert isinstance(enhanced_feedback, str)
    assert len(enhanced_feedback) > 0
    assert enhanced_feedback != base_feedback  # 強化されていることを確認

def test_different_emotion_states(feedback_generator, mock_learning_state,
                                mock_task_history):
    """異なる感情状態でのフィードバック生成テスト"""
    emotions = [
        EmotionCategory.MOTIVATED,
        EmotionCategory.FRUSTRATED,
        EmotionCategory.CONFUSED,
        EmotionCategory.SATISFIED
    ]
    
    for emotion in emotions:
        emotion_result = EmotionAnalysisResult(
            primary_emotion=emotion,
            emotion_scores={e: 0.1 for e in EmotionCategory},
            confidence=0.8
        )
        
        context = FeedbackContext(
            emotion_state=emotion_result,
            learning_state=mock_learning_state,
            task_history=mock_task_history,
            preferred_style=None
        )
        
        feedback = feedback_generator.generate_feedback(context)
        assert isinstance(feedback, str)
        assert len(feedback) > 0

def test_different_feedback_styles(feedback_generator, mock_emotion_result,
                                 mock_learning_state, mock_task_history):
    """異なるフィードバックスタイルのテスト"""
    for style in FeedbackStyle:
        context = FeedbackContext(
            emotion_state=mock_emotion_result,
            learning_state=mock_learning_state,
            task_history=mock_task_history,
            preferred_style=style
        )
        
        feedback = feedback_generator.generate_feedback(context)
        assert isinstance(feedback, str)
        assert len(feedback) > 0

def test_empty_task_history(feedback_generator, mock_emotion_result,
                          mock_learning_state):
    """空のタスク履歴でのフィードバック生成テスト"""
    context = FeedbackContext(
        emotion_state=mock_emotion_result,
        learning_state=mock_learning_state,
        task_history=[],
        preferred_style=None
    )
    
    feedback = feedback_generator.generate_feedback(context)
    assert isinstance(feedback, str)
    assert len(feedback) > 0

def test_extreme_learning_states(feedback_generator, mock_emotion_result,
                               mock_task_history):
    """極端な学習状態でのフィードバック生成テスト"""
    # 非常に高い成功率
    high_performance_state = {
        'current_level': 9,
        'success_rate': 0.95,
        'challenge_count': 100,
        'focus_duration': 90,
        'help_requests': 0
    }
    
    # 非常に低い成功率
    low_performance_state = {
        'current_level': 1,
        'success_rate': 0.2,
        'challenge_count': 5,
        'focus_duration': 15,
        'help_requests': 10
    }
    
    for state in [high_performance_state, low_performance_state]:
        context = FeedbackContext(
            emotion_state=mock_emotion_result,
            learning_state=state,
            task_history=mock_task_history,
            preferred_style=None
        )
        
        feedback = feedback_generator.generate_feedback(context)
        assert isinstance(feedback, str)
        assert len(feedback) > 0
