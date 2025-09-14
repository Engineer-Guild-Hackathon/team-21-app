"""
会話履歴から非認知能力を分析するMLモデル
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


@dataclass
class NonCognitiveSkills:
    """非認知能力スコア"""

    grit: float  # やり抜く力 (0.0-5.0)
    collaboration: float  # 協調性 (0.0-5.0)
    self_regulation: float  # 自己制御 (0.0-5.0)
    emotional_intelligence: float  # 感情知能 (0.0-5.0)
    confidence: float = 0.0  # 自信 (0.0-5.0)


@dataclass
class ConversationMetrics:
    """会話のメトリクス"""

    message_count: int
    avg_message_length: float
    question_count: int
    positive_words_count: int
    negative_words_count: int
    learning_keywords_count: int
    help_seeking_count: int
    completion_rate: float  # 問題解決完了率


class ConversationAnalyzer:
    """会話履歴から非認知能力を分析するクラス"""

    def __init__(self):
        self.grit_model = None
        self.collaboration_model = None
        self.self_regulation_model = None
        self.emotional_intelligence_model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self.scaler = StandardScaler()
        self._load_models()

    def _load_models(self):
        """事前訓練済みモデルを読み込む"""
        models_dir = os.path.join(os.path.dirname(__file__), "trained_models")

        if os.path.exists(models_dir):
            try:
                self.grit_model = joblib.load(
                    os.path.join(models_dir, "grit_model.pkl")
                )
                self.collaboration_model = joblib.load(
                    os.path.join(models_dir, "collaboration_model.pkl")
                )
                self.self_regulation_model = joblib.load(
                    os.path.join(models_dir, "self_regulation_model.pkl")
                )
                self.emotional_intelligence_model = joblib.load(
                    os.path.join(models_dir, "emotional_intelligence_model.pkl")
                )
                self.vectorizer = joblib.load(
                    os.path.join(models_dir, "vectorizer.pkl")
                )
                self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
            except FileNotFoundError:
                print(
                    "事前訓練済みモデルが見つかりません。デフォルトモデルを使用します。"
                )
                self._create_default_models()
        else:
            self._create_default_models()

    def _create_default_models(self):
        """デフォルトのルールベースモデルを作成"""
        self.grit_model = self._create_rule_based_model()
        self.collaboration_model = self._create_rule_based_model()
        self.self_regulation_model = self._create_rule_based_model()
        self.emotional_intelligence_model = self._create_rule_based_model()

    def _create_rule_based_model(self):
        """ルールベースのダミーモデル"""

        class RuleBasedModel:
            def predict(self, X):
                return np.random.uniform(1.0, 4.0, len(X))

        return RuleBasedModel()

    def analyze_conversation(self, messages: List[Dict]) -> NonCognitiveSkills:
        """会話履歴を分析して非認知能力スコアを算出"""

        # 会話メトリクスを計算
        metrics = self._calculate_conversation_metrics(messages)

        # テキストデータを準備
        conversation_text = self._extract_conversation_text(messages)

        # 特徴量を抽出
        features = self._extract_features(conversation_text, metrics)

        # 会話内容に基づく個別調整
        content_adjustments = self._analyze_conversation_content(messages)

        # 各非認知能力スコアを予測（内容調整を適用）
        grit_score = self._predict_grit(features) + content_adjustments.get("grit", 0)
        collaboration_score = self._predict_collaboration(
            features
        ) + content_adjustments.get("collaboration", 0)
        self_regulation_score = self._predict_self_regulation(
            features
        ) + content_adjustments.get("self_regulation", 0)
        emotional_intelligence_score = self._predict_emotional_intelligence(
            features
        ) + content_adjustments.get("emotional_intelligence", 0)
        confidence_score = self._predict_confidence(features) + content_adjustments.get(
            "confidence", 0
        )

        # スコアを0.0-5.0の範囲に制限
        grit_score = max(0.0, min(5.0, grit_score))
        collaboration_score = max(0.0, min(5.0, collaboration_score))
        self_regulation_score = max(0.0, min(5.0, self_regulation_score))
        emotional_intelligence_score = max(0.0, min(5.0, emotional_intelligence_score))
        confidence_score = max(0.0, min(5.0, confidence_score))

        return NonCognitiveSkills(
            grit=grit_score,
            collaboration=collaboration_score,
            self_regulation=self_regulation_score,
            emotional_intelligence=emotional_intelligence_score,
            confidence=confidence_score,
        )

    def analyze_conversation_with_context(
        self, messages: List[Dict]
    ) -> NonCognitiveSkills:
        """会話履歴を文脈を含めて分析（データベースベース分析用）"""
        if not messages:
            return NonCognitiveSkills()

        # 時系列で会話を分析し、成長の軌跡を追跡
        user_messages = [msg for msg in messages if msg.get("role") == "user"]

        # 基本的な分析を実行
        basic_skills = self.analyze_conversation(messages)

        # 文脈を考慮した補正を適用
        context_adjustments = self._calculate_context_adjustments(messages)

        # スキルに文脈補正を適用
        adjusted_skills = NonCognitiveSkills(
            grit=basic_skills.grit + context_adjustments.get("grit", 0),
            collaboration=basic_skills.collaboration
            + context_adjustments.get("collaboration", 0),
            self_regulation=basic_skills.self_regulation
            + context_adjustments.get("self_regulation", 0),
            emotional_intelligence=basic_skills.emotional_intelligence
            + context_adjustments.get("emotional_intelligence", 0),
            confidence=basic_skills.confidence
            + context_adjustments.get("confidence", 0),
        )

        # スコアを0.0-5.0の範囲に制限
        return NonCognitiveSkills(
            grit=max(0.0, min(5.0, adjusted_skills.grit)),
            collaboration=max(0.0, min(5.0, adjusted_skills.collaboration)),
            self_regulation=max(0.0, min(5.0, adjusted_skills.self_regulation)),
            emotional_intelligence=max(
                0.0, min(5.0, adjusted_skills.emotional_intelligence)
            ),
            confidence=max(0.0, min(5.0, adjusted_skills.confidence)),
        )

    def _calculate_context_adjustments(self, messages: List[Dict]) -> Dict[str, float]:
        """文脈を考慮したスキル補正値を計算"""
        adjustments = {
            "grit": 0.0,
            "collaboration": 0.0,
            "self_regulation": 0.0,
            "emotional_intelligence": 0.0,
            "confidence": 0.0,
        }

        if len(messages) < 2:
            return adjustments

        # 会話の継続性を評価（長期間の継続的な学習を示す）
        if len(messages) > 10:
            adjustments["grit"] += 0.3  # 継続性
            adjustments["self_regulation"] += 0.2  # 自己管理

        # 質問の質と深さを評価
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        question_quality = self._evaluate_question_quality(user_messages)

        if question_quality > 0.7:
            adjustments["confidence"] += 0.2  # 自信
            adjustments["emotional_intelligence"] += 0.1  # 感情理解

        # 学習トピックの多様性を評価
        topic_diversity = self._evaluate_topic_diversity(user_messages)
        if topic_diversity > 0.6:
            adjustments["collaboration"] += 0.2  # 協調性
            adjustments["emotional_intelligence"] += 0.1  # 感情理解

        return adjustments

    def _evaluate_question_quality(self, user_messages: List[Dict]) -> float:
        """ユーザーの質問の質を評価"""
        if not user_messages:
            return 0.0

        quality_indicators = [
            "なぜ",
            "どのように",
            "なぜなら",
            "例えば",
            "具体的に",
            "違いは",
            "関係は",
            "意味は",
            "理由は",
            "方法は",
        ]

        total_questions = len(user_messages)
        quality_questions = 0

        for msg in user_messages:
            content = msg.get("content", "").lower()
            if any(indicator in content for indicator in quality_indicators):
                quality_questions += 1

        return quality_questions / total_questions if total_questions > 0 else 0.0

    def _evaluate_topic_diversity(self, user_messages: List[Dict]) -> float:
        """学習トピックの多様性を評価"""
        if not user_messages:
            return 0.0

        topics = set()
        topic_keywords = {
            "数学": ["数学", "算数", "計算", "方程式", "幾何", "代数"],
            "国語": ["国語", "日本語", "文章", "読解", "作文", "文法"],
            "理科": ["理科", "科学", "実験", "物理", "化学", "生物"],
            "社会": ["社会", "歴史", "地理", "政治", "経済", "文化"],
            "英語": ["英語", "英会話", "単語", "文法", "リスニング"],
            "芸術": ["芸術", "音楽", "美術", "絵画", "演奏"],
            "体育": ["体育", "運動", "スポーツ", "健康", "体力"],
        }

        for msg in user_messages:
            content = msg.get("content", "").lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in content for keyword in keywords):
                    topics.add(topic)

        return len(topics) / len(topic_keywords)

    def _calculate_conversation_metrics(
        self, messages: List[Dict]
    ) -> ConversationMetrics:
        """会話のメトリクスを計算"""
        user_messages = [msg for msg in messages if msg.get("role") == "user"]

        total_messages = len(user_messages)
        total_length = sum(len(msg.get("content", "")) for msg in user_messages)
        avg_length = total_length / max(total_messages, 1)

        # 質問の数をカウント
        question_count = sum(
            1
            for msg in user_messages
            if "?" in msg.get("content", "")
            or "ですか" in msg.get("content", "")
            or "でしょうか" in msg.get("content", "")
        )

        # ポジティブ・ネガティブワードをカウント
        positive_words = [
            "頑張る",
            "努力",
            "続ける",
            "挑戦",
            "学ぶ",
            "理解",
            "できた",
            "成功",
            "嬉しい",
            "楽しい",
        ]
        negative_words = [
            "難しい",
            "わからない",
            "できない",
            "困った",
            "疲れた",
            "嫌い",
            "苦手",
            "失敗",
        ]

        all_text = " ".join(msg.get("content", "") for msg in user_messages)
        positive_count = sum(all_text.count(word) for word in positive_words)
        negative_count = sum(all_text.count(word) for word in negative_words)

        # 学習関連キーワードをカウント
        learning_keywords = [
            "勉強",
            "学習",
            "問題",
            "課題",
            "宿題",
            "授業",
            "教科書",
            "テスト",
            "試験",
        ]
        learning_keywords_count = sum(
            all_text.count(word) for word in learning_keywords
        )

        # ヘルプシーキングをカウント
        help_words = [
            "教えて",
            "助けて",
            "手伝って",
            "わからない",
            "困っている",
            "どうすれば",
        ]
        help_seeking_count = sum(all_text.count(word) for word in help_words)

        # 完了率（簡易版：質問に対する返答の有無）
        completion_rate = min(
            1.0, total_messages / max(1, len(messages) - total_messages)
        )

        return ConversationMetrics(
            message_count=total_messages,
            avg_message_length=avg_length,
            question_count=question_count,
            positive_words_count=positive_count,
            negative_words_count=negative_count,
            learning_keywords_count=learning_keywords_count,
            help_seeking_count=help_seeking_count,
            completion_rate=completion_rate,
        )

    def _extract_conversation_text(self, messages: List[Dict]) -> str:
        """会話からテキストを抽出"""
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        return " ".join(msg.get("content", "") for msg in user_messages)

    def _extract_features(
        self, conversation_text: str, metrics: ConversationMetrics
    ) -> np.ndarray:
        """特徴量を抽出"""
        features = [
            metrics.message_count,
            metrics.avg_message_length,
            metrics.question_count,
            metrics.positive_words_count,
            metrics.negative_words_count,
            metrics.learning_keywords_count,
            metrics.help_seeking_count,
            metrics.completion_rate,
            len(conversation_text.split()),
            len(re.findall(r"[!！]", conversation_text)),  # 感嘆符の数
            len(re.findall(r"[?？]", conversation_text)),  # 疑問符の数
        ]

        return np.array(features).reshape(1, -1)

    def _predict_grit(self, features: np.ndarray) -> float:
        """グリット（やり抜く力）を予測"""
        if self.grit_model:
            score = self.grit_model.predict(features)[0]
        else:
            # ルールベースの簡易予測
            score = min(
                5.0, max(1.0, 2.0 + features[0][2] * 0.1 + features[0][3] * 0.2)
            )
        return round(score, 2)

    def _predict_collaboration(self, features: np.ndarray) -> float:
        """協調性を予測"""
        if self.collaboration_model:
            score = self.collaboration_model.predict(features)[0]
        else:
            # ルールベースの簡易予測
            score = min(
                5.0, max(1.0, 2.0 + features[0][6] * 0.3 + features[0][9] * 0.1)
            )
        return round(score, 2)

    def _predict_self_regulation(self, features: np.ndarray) -> float:
        """自己制御を予測"""
        if self.self_regulation_model:
            score = self.self_regulation_model.predict(features)[0]
        else:
            # ルールベースの簡易予測
            score = min(
                5.0, max(1.0, 2.0 + features[0][1] * 0.1 + features[0][7] * 0.4)
            )
        return round(score, 2)

    def _predict_emotional_intelligence(self, features: np.ndarray) -> float:
        """感情知能を予測"""
        if self.emotional_intelligence_model:
            score = self.emotional_intelligence_model.predict(features)[0]
        else:
            # ルールベースの簡易予測
            score = min(
                5.0, max(1.0, 2.0 + features[0][3] * 0.2 - features[0][4] * 0.1)
            )
        return round(score, 2)

    def _predict_confidence(self, features: np.ndarray) -> float:
        """自信を予測"""
        # ルールベースの簡易予測
        score = min(5.0, max(1.0, 2.0 + features[0][3] * 0.2 + features[0][7] * 0.3))
        return round(score, 2)

    def generate_feedback(
        self,
        skills: NonCognitiveSkills,
        previous_skills: Optional[NonCognitiveSkills] = None,
    ) -> str:
        """非認知能力スコアに基づいて個別化されたフィードバックを生成"""

        import random

        feedback_parts = []

        # より多様なフィードバックテンプレート
        grit_feedbacks = {
            "high": [
                "🌟 素晴らしいやり抜く力！困難な課題にも諦めずに取り組む姿勢が印象的です。",
                "💪 継続力が抜群ですね！目標に向かって一歩一歩進む姿勢が素晴らしいです。",
                "🏆 粘り強さが際立っています！どんな困難も乗り越えられる力を持っていますね。",
                "🎯 目標達成への意志が強く感じられます！この調子で頑張ってください。",
            ],
            "medium": [
                "👍 やり抜く力が育っています！もう少し継続することで大きな成果が得られるでしょう。",
                "📈 努力する姿勢が見えてきました！小さな成功を積み重ねていきましょう。",
                "🔄 継続性が向上しています！習慣化することでさらに力を伸ばせます。",
                "⚡ 集中力が高まってきています！この勢いで学習を続けてみてください。",
            ],
            "low": [
                "💡 やり抜く力を育てるために、まずは小さな目標から始めてみましょう。",
                "🎪 楽しく継続できる方法を見つけて、学習を習慣にしていきましょう。",
                "🌱 成長の種をまいている段階です！焦らずに一つずつ取り組んでみてください。",
                "🚀 スタートラインに立っています！小さな一歩から始めてみませんか？",
            ],
        }

        collaboration_feedbacks = {
            "high": [
                "🤝 協調性が素晴らしいです！他者との協力を大切にする姿勢が印象的です。",
                "👥 チームワークの才能があります！みんなと一緒に学ぶ楽しさを感じていますね。",
                "🤲 思いやりのある学習姿勢が素晴らしいです！周りの人も支えているでしょう。",
                "🎭 コミュニケーション能力が高いですね！積極的に交流する姿勢が良いです。",
            ],
            "medium": [
                "👂 協調性が育っています！相手の意見に耳を傾ける姿勢が見えてきました。",
                "💬 コミュニケーションが向上しています！質問や相談を積極的にしてみましょう。",
                "🤝 協力する姿勢が見えてきました！グループ学習でさらに力を伸ばせます。",
                "👥 社交性が高まっています！友達と一緒に学習する機会を増やしてみましょう。",
            ],
            "low": [
                "🗣️ 協調性を高めるために、まずは質問や相談を積極的にしてみましょう。",
                "👥 グループ学習に参加して、他の人との交流を楽しんでみてください。",
                "💭 自分の意見を伝える練習から始めて、協調性を育てていきましょう。",
                "🤲 他者への思いやりを意識して、協力的な姿勢を身につけていきましょう。",
            ],
        }

        self_regulation_feedbacks = {
            "high": [
                "🎯 自己管理能力が優れています！計画的に学習を進められていますね。",
                "⏰ 時間管理が素晴らしいです！効率的な学習スタイルが身についています。",
                "📋 計画性が抜群ですね！目標に向かって着実に進んでいます。",
                "🧠 集中力と自制心が高いです！学習に取り組む姿勢が素晴らしいです。",
            ],
            "medium": [
                "📝 自己管理力が向上しています！学習計画を立てて実行してみましょう。",
                "⏱️ 時間の使い方が改善されてきました！さらに効率化を図ってみてください。",
                "📊 計画性が育っています！目標設定を明確にして取り組んでみましょう。",
                "🎪 集中力が高まってきています！学習環境を整えてさらに力を伸ばしましょう。",
            ],
            "low": [
                "📅 自己管理力を高めるために、まずは学習時間を決めて取り組んでみましょう。",
                "🎯 目標設定から始めて、計画的に学習を進めていきましょう。",
                "⏰ 時間管理の練習をして、効率的な学習スタイルを身につけましょう。",
                "🧘 集中力を高めるために、学習環境を整えて取り組んでみましょう。",
            ],
        }

        emotional_intelligence_feedbacks = {
            "high": [
                "💝 感情知能が高いです！自分の感情を理解し、適切に表現できています。",
                "😊 感情のコントロールが素晴らしいです！安定した学習姿勢が印象的です。",
                "🌈 感情の豊かさと表現力が優れています！学習にも良い影響を与えていますね。",
                "🤗 共感力が高く、他者との関係性も良好ですね！学習環境も良くなっているでしょう。",
            ],
            "medium": [
                "😌 感情知能が育っています！感情を言葉で表現する練習をしてみましょう。",
                "🧘 感情の安定性が向上しています！ストレス管理も意識してみてください。",
                "💭 自己理解が深まってきました！感情を振り返る時間を作ってみましょう。",
                "🎭 感情表現が豊かになってきました！学習へのモチベーションも高まっているでしょう。",
            ],
            "low": [
                "💭 感情知能を高めるために、自分の気持ちを振り返る時間を作ってみましょう。",
                "😊 ポジティブな感情を意識して、学習へのモチベーションを高めていきましょう。",
                "🧠 感情と思考のバランスを取って、安定した学習姿勢を身につけましょう。",
                "🌈 感情の表現力を高めて、学習への意欲を育てていきましょう。",
            ],
        }

        # スキルレベルに応じてフィードバックを選択
        def get_feedback_level(score):
            if score >= 4.0:
                return "high"
            elif score >= 3.0:
                return "medium"
            else:
                return "low"

        # 各スキルのフィードバックを生成（ランダム選択）
        grit_level = get_feedback_level(skills.grit)
        feedback_parts.append(random.choice(grit_feedbacks[grit_level]))

        collaboration_level = get_feedback_level(skills.collaboration)
        feedback_parts.append(
            random.choice(collaboration_feedbacks[collaboration_level])
        )

        self_regulation_level = get_feedback_level(skills.self_regulation)
        feedback_parts.append(
            random.choice(self_regulation_feedbacks[self_regulation_level])
        )

        emotional_intelligence_level = get_feedback_level(skills.emotional_intelligence)
        feedback_parts.append(
            random.choice(
                emotional_intelligence_feedbacks[emotional_intelligence_level]
            )
        )

        # 進歩の評価
        if previous_skills:
            improvements = []
            if skills.grit > previous_skills.grit + 0.2:
                improvements.append("やり抜く力")
            if skills.collaboration > previous_skills.collaboration + 0.2:
                improvements.append("協調性")
            if skills.self_regulation > previous_skills.self_regulation + 0.2:
                improvements.append("自己制御力")
            if (
                skills.emotional_intelligence
                > previous_skills.emotional_intelligence + 0.2
            ):
                improvements.append("感情知能")

            if improvements:
                feedback_parts.append(
                    f"🎉 素晴らしい進歩です！{', '.join(improvements)}が向上しています。"
                )

        return "\n\n".join(feedback_parts)

    def _analyze_conversation_content(self, messages: List[Dict]) -> Dict[str, float]:
        """会話内容に基づいてスキル調整値を計算"""

        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        all_text = " ".join(msg.get("content", "") for msg in user_messages).lower()

        adjustments = {
            "grit": 0.0,
            "collaboration": 0.0,
            "self_regulation": 0.0,
            "emotional_intelligence": 0.0,
            "confidence": 0.0,
        }

        # グリット関連のキーワード
        grit_positive = [
            "頑張る",
            "続ける",
            "挑戦",
            "努力",
            "目標",
            "達成",
            "やり抜く",
            "諦めない",
        ]
        grit_negative = ["諦める", "やめる", "面倒", "疲れた", "無理"]

        grit_score = sum(1 for word in grit_positive if word in all_text) * 0.1
        grit_score -= sum(1 for word in grit_negative if word in all_text) * 0.05
        adjustments["grit"] = grit_score

        # 協調性関連のキーワード
        collaboration_positive = [
            "一緒",
            "協力",
            "助ける",
            "質問",
            "相談",
            "教える",
            "グループ",
            "チーム",
        ]
        collaboration_negative = ["一人", "独り", "自分だけ", "他人", "邪魔"]

        collab_score = (
            sum(1 for word in collaboration_positive if word in all_text) * 0.1
        )
        collab_score -= (
            sum(1 for word in collaboration_negative if word in all_text) * 0.05
        )
        adjustments["collaboration"] = collab_score

        # 自己制御関連のキーワード
        regulation_positive = ["計画", "時間", "集中", "計画的", "整理", "管理", "習慣"]
        regulation_negative = ["だらだら", "散漫", "集中できない", "計画なし"]

        reg_score = sum(1 for word in regulation_positive if word in all_text) * 0.1
        reg_score -= sum(1 for word in regulation_negative if word in all_text) * 0.05
        adjustments["self_regulation"] = reg_score

        # 感情知能関連のキーワード
        emotion_positive = [
            "嬉しい",
            "楽しい",
            "感情",
            "気持ち",
            "理解",
            "共感",
            "感謝",
        ]
        emotion_negative = ["怒る", "イライラ", "悲しい", "不安", "ストレス"]

        emotion_score = sum(1 for word in emotion_positive if word in all_text) * 0.1
        emotion_score -= sum(1 for word in emotion_negative if word in all_text) * 0.03
        adjustments["emotional_intelligence"] = emotion_score

        # 自信関連のキーワード
        confidence_positive = ["できる", "大丈夫", "自信", "成功", "得意", "好き"]
        confidence_negative = ["できない", "無理", "自信ない", "苦手", "嫌い"]

        conf_score = sum(1 for word in confidence_positive if word in all_text) * 0.1
        conf_score -= sum(1 for word in confidence_negative if word in all_text) * 0.05
        adjustments["confidence"] = conf_score

        # 調整値を-0.5から+0.5の範囲に制限
        for key in adjustments:
            adjustments[key] = max(-0.5, min(0.5, adjustments[key]))

        return adjustments
