import numpy as np
import pytest
import torch

from ..models.emotion_analyzer import (EmotionAnalyzer, EmotionCategory,
                                       EmotionClassifier, EmotionTracker)


@pytest.fixture
def emotion_analyzer():
    return EmotionAnalyzer()


@pytest.fixture
def emotion_tracker():
    return EmotionTracker(window_size=5)


def test_emotion_analyzer_initialization(emotion_analyzer):
    """感情分析モジュールの初期化テスト"""
    assert emotion_analyzer.bert_model is not None
    assert emotion_analyzer.tokenizer is not None
    assert emotion_analyzer.emotion_classifier is not None
    assert isinstance(emotion_analyzer.emotion_classifier, EmotionClassifier)


def test_text_feature_extraction(emotion_analyzer):
    """テキスト特徴抽出のテスト"""
    test_text = "この問題は難しいけど、頑張って解いてみます。"
    features = emotion_analyzer._extract_text_features(test_text)

    assert isinstance(features, torch.Tensor)
    assert features.shape[1] == 768  # BERT hidden size
    assert torch.isfinite(features).all()


def test_behavioral_feature_extraction(emotion_analyzer):
    """行動特徴抽出のテスト"""
    behavioral_data = {
        "challenge_attempts": 5,
        "success_rate": 0.7,
        "response_time": 30,
        "help_requests": 2,
        "task_switches": 3,
        "focus_duration": 15,
        "error_rate": 0.2,
        "correction_rate": 0.8,
        "exploration_rate": 0.5,
        "completion_rate": 0.6,
    }

    features = emotion_analyzer._extract_behavioral_features(behavioral_data)

    assert isinstance(features, torch.Tensor)
    assert features.shape[0] == 10  # 行動特徴の次元数
    assert torch.isfinite(features).all()


def test_emotion_analysis(emotion_analyzer):
    """感情分析の統合テスト"""
    test_text = "わからないところがあって困っています。"
    behavioral_data = {
        "challenge_attempts": 3,
        "success_rate": 0.4,
        "response_time": 45,
        "help_requests": 2,
        "task_switches": 1,
        "focus_duration": 20,
        "error_rate": 0.3,
        "correction_rate": 0.6,
        "exploration_rate": 0.4,
        "completion_rate": 0.5,
    }

    result = emotion_analyzer.analyze_emotion(test_text, behavioral_data)

    assert result.primary_emotion in EmotionCategory
    assert isinstance(result.emotion_scores, dict)
    assert isinstance(result.confidence, float)
    assert 0 <= result.confidence <= 1
    assert all(isinstance(score, float) for score in result.emotion_scores.values())


def test_emotion_classifier(emotion_analyzer):
    """感情分類器のテスト"""
    classifier = emotion_analyzer.emotion_classifier

    # テスト用の入力テンソルを作成
    text_features = torch.randn(1, 768)
    behavioral_features = torch.randn(1, 10)

    # 順伝播のテスト
    output = classifier(text_features, behavioral_features)

    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == len(EmotionCategory)
    assert torch.isfinite(output).all()


def test_emotion_tracker_basic(emotion_tracker):
    """感情トラッカーの基本機能テスト"""
    # テスト用の感情分析結果を作成
    from dataclasses import dataclass

    @dataclass
    class MockEmotionResult:
        primary_emotion: EmotionCategory
        emotion_scores: dict
        confidence: float

    result = MockEmotionResult(
        primary_emotion=EmotionCategory.MOTIVATED,
        emotion_scores={emotion: 0.1 for emotion in EmotionCategory},
        confidence=0.8,
    )

    # 結果の追加
    emotion_tracker.add_emotion(result)

    assert len(emotion_tracker.emotion_history) == 1
    assert (
        emotion_tracker.emotion_history[0].primary_emotion == EmotionCategory.MOTIVATED
    )


def test_emotion_trend_analysis(emotion_tracker):
    """感情傾向分析のテスト"""
    from dataclasses import dataclass

    @dataclass
    class MockEmotionResult:
        primary_emotion: EmotionCategory
        emotion_scores: dict
        confidence: float

    # 複数の感情結果を追加
    emotions = [
        EmotionCategory.MOTIVATED,
        EmotionCategory.CONFUSED,
        EmotionCategory.CONFIDENT,
    ]
    for emotion in emotions:
        result = MockEmotionResult(
            primary_emotion=emotion,
            emotion_scores={e: 0.1 for e in EmotionCategory},
            confidence=0.8,
        )
        emotion_tracker.add_emotion(result)

    trend = emotion_tracker.get_emotion_trend()

    assert isinstance(trend, dict)
    assert all(isinstance(score, float) for score in trend.values())


def test_behavioral_trend_analysis(emotion_tracker):
    """行動傾向分析のテスト"""
    from dataclasses import dataclass

    @dataclass
    class MockEmotionResult:
        primary_emotion: EmotionCategory
        emotion_scores: dict
        confidence: float

    # 感情結果と行動データを追加
    behavioral_data = {
        "challenge_attempts": 5,
        "success_rate": 0.7,
        "focus_duration": 30,
    }

    result = MockEmotionResult(
        primary_emotion=EmotionCategory.MOTIVATED,
        emotion_scores={emotion: 0.1 for emotion in EmotionCategory},
        confidence=0.8,
    )

    emotion_tracker.add_emotion(result, behavioral_data)
    trend = emotion_tracker.get_behavioral_trend()

    assert isinstance(trend, dict)
    assert all(isinstance(value, float) for value in trend.values())
    assert "challenge_attempts" in trend
    assert "success_rate" in trend


def test_emotional_change_detection(emotion_tracker):
    """感情変化の検出テスト"""
    from dataclasses import dataclass

    @dataclass
    class MockEmotionResult:
        primary_emotion: EmotionCategory
        emotion_scores: dict
        confidence: float

    # 異なる感情状態を追加
    emotions = [
        EmotionCategory.MOTIVATED,
        EmotionCategory.MOTIVATED,
        EmotionCategory.CONFUSED,
        EmotionCategory.CONFIDENT,
    ]

    for emotion in emotions:
        result = MockEmotionResult(
            primary_emotion=emotion,
            emotion_scores={e: 0.1 for e in EmotionCategory},
            confidence=0.8,
        )
        emotion_tracker.add_emotion(result)

    changes = emotion_tracker.detect_emotional_changes()

    assert isinstance(changes, list)
    assert len(changes) == 2  # 2回の感情変化
    assert all(isinstance(change, dict) for change in changes)
    assert all(
        "from_emotion" in change and "to_emotion" in change for change in changes
    )


def test_json_serialization(emotion_tracker):
    """JSON シリアライズのテスト"""
    from dataclasses import dataclass

    @dataclass
    class MockEmotionResult:
        primary_emotion: EmotionCategory
        emotion_scores: dict
        confidence: float

    # データの追加
    result = MockEmotionResult(
        primary_emotion=EmotionCategory.MOTIVATED,
        emotion_scores={emotion: 0.1 for emotion in EmotionCategory},
        confidence=0.8,
    )

    behavioral_data = {"challenge_attempts": 5, "success_rate": 0.7}

    emotion_tracker.add_emotion(result, behavioral_data)
    json_output = emotion_tracker.to_json()

    assert isinstance(json_output, str)
    assert "primary_emotion" in json_output
    assert "confidence" in json_output
    assert "behavioral_data" in json_output
