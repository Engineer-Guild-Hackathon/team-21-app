"""
MLサービス用のFastAPIアプリケーション
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..models.conversation_analyzer import (ConversationAnalyzer,
                                            NonCognitiveSkills)
from ..models.progress_predictor import (LearningActivity, ProgressPrediction,
                                         ProgressPredictor)

app = FastAPI(
    title="NonCog ML Service",
    description="非認知能力学習プラットフォーム用MLサービス",
    version="1.0.0",
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# バックエンドAPIのベースURL
BACKEND_API_BASE = os.getenv("BACKEND_API_BASE", "http://localhost:8000")


# リクエスト/レスポンスモデル
class Message(BaseModel):
    id: str
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: datetime


class ConversationAnalysisRequest(BaseModel):
    user_id: int
    messages: List[Message]
    current_skills: Optional[Dict[str, float]] = None


class ConversationAnalysisResponse(BaseModel):
    user_id: int
    skills: NonCognitiveSkills
    feedback: str
    analysis_timestamp: datetime


class ProgressUpdateRequest(BaseModel):
    user_id: int
    activities: List[Dict]  # 学習活動データ
    current_skills: Dict[str, float]
    time_horizon_days: int = 7


class ProgressUpdateResponse(BaseModel):
    user_id: int
    prediction: ProgressPrediction
    updated_skills: Dict[str, float]
    update_timestamp: datetime


class FeedbackRequest(BaseModel):
    user_id: int
    message: str
    context: Optional[Dict] = None


class FeedbackResponse(BaseModel):
    user_id: int
    feedback: str
    suggestions: List[str]
    timestamp: datetime


# MLモデルの初期化
conversation_analyzer = ConversationAnalyzer()
progress_predictor = ProgressPredictor()


async def get_backend_client():
    """バックエンドAPI用のHTTPクライアント"""
    async with httpx.AsyncClient() as client:
        yield client


@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "NonCog ML Service is running"}


@app.post("/analyze-conversation", response_model=ConversationAnalysisResponse)
async def analyze_conversation(
    request: ConversationAnalysisRequest,
    client: httpx.AsyncClient = Depends(get_backend_client),
):
    """会話履歴を分析して非認知能力スコアを算出"""

    try:
        # メッセージを変換
        messages = [
            {
                "id": msg.id,
                "content": msg.content,
                "role": msg.role,
                "timestamp": msg.timestamp,
            }
            for msg in request.messages
        ]

        # 会話分析を実行
        skills = conversation_analyzer.analyze_conversation(messages)

        # 以前のスキルと比較してフィードバックを生成
        previous_skills = None
        if request.current_skills:
            previous_skills = NonCognitiveSkills(
                grit=request.current_skills.get("grit", 2.0),
                collaboration=request.current_skills.get("collaboration", 2.0),
                self_regulation=request.current_skills.get("self_regulation", 2.0),
                emotional_intelligence=request.current_skills.get(
                    "emotional_intelligence", 2.0
                ),
                confidence=request.current_skills.get("confidence", 2.0),
            )

        feedback = conversation_analyzer.generate_feedback(skills, previous_skills)

        # バックエンドのユーザー統計を更新
        await update_user_stats(client, request.user_id, skills)

        return ConversationAnalysisResponse(
            user_id=request.user_id,
            skills=skills,
            feedback=feedback,
            analysis_timestamp=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"会話分析エラー: {str(e)}")


@app.post("/predict-progress", response_model=ProgressUpdateResponse)
async def predict_progress(
    request: ProgressUpdateRequest,
    client: httpx.AsyncClient = Depends(get_backend_client),
):
    """学習行動から進捗を予測してユーザー統計を更新"""

    try:
        # 学習活動データを変換
        activities = []
        for activity_data in request.activities:
            activity = LearningActivity(
                timestamp=datetime.fromisoformat(
                    activity_data["timestamp"].replace("Z", "+00:00")
                ),
                activity_type=activity_data["activity_type"],
                duration_minutes=activity_data["duration_minutes"],
                success_rate=activity_data["success_rate"],
                difficulty_level=activity_data["difficulty_level"],
                engagement_score=activity_data["engagement_score"],
            )
            activities.append(activity)

        # 進捗予測を実行
        prediction = progress_predictor.predict_progress(
            activities, request.current_skills, request.time_horizon_days
        )

        # 更新されたスキルを計算
        updated_skills = {
            "grit": min(
                5.0,
                request.current_skills.get("grit", 2.0) + prediction.grit_improvement,
            ),
            "collaboration": min(
                5.0,
                request.current_skills.get("collaboration", 2.0)
                + prediction.collaboration_improvement,
            ),
            "self_regulation": min(
                5.0,
                request.current_skills.get("self_regulation", 2.0)
                + prediction.self_regulation_improvement,
            ),
            "emotional_intelligence": min(
                5.0,
                request.current_skills.get("emotional_intelligence", 2.0)
                + prediction.emotional_intelligence_improvement,
            ),
            "confidence": min(
                5.0,
                request.current_skills.get("confidence", 2.0)
                + prediction.confidence_improvement,
            ),
        }

        # バックエンドのユーザー統計を更新
        await update_user_stats(client, request.user_id, None, updated_skills)

        return ProgressUpdateResponse(
            user_id=request.user_id,
            prediction=prediction,
            updated_skills=updated_skills,
            update_timestamp=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"進捗予測エラー: {str(e)}")


@app.post("/generate-feedback", response_model=FeedbackResponse)
async def generate_feedback(request: FeedbackRequest):
    """リアルタイムフィードバック生成"""

    try:
        # 簡易的なフィードバック生成（実際の実装ではより高度なNLPを使用）
        feedback = generate_contextual_feedback(request.message, request.context)
        suggestions = generate_suggestions(request.message, request.context)

        return FeedbackResponse(
            user_id=request.user_id,
            feedback=feedback,
            suggestions=suggestions,
            timestamp=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"フィードバック生成エラー: {str(e)}"
        )


async def update_user_stats(
    client: httpx.AsyncClient,
    user_id: int,
    skills: Optional[NonCognitiveSkills] = None,
    updated_skills: Optional[Dict[str, float]] = None,
):
    """バックエンドのユーザー統計を更新"""

    try:
        # ユーザー統計の更新データを準備
        update_data = {}

        if skills:
            update_data.update(
                {
                    "grit_level": skills.grit,
                    "collaboration_level": skills.collaboration,
                    "self_regulation_level": skills.self_regulation,
                    "emotional_intelligence_level": skills.emotional_intelligence,
                }
            )

        if updated_skills:
            update_data.update(
                {
                    "grit_level": updated_skills["grit"],
                    "collaboration_level": updated_skills["collaboration"],
                    "self_regulation_level": updated_skills["self_regulation"],
                    "emotional_intelligence_level": updated_skills[
                        "emotional_intelligence"
                    ],
                }
            )

        # バックエンドAPIを呼び出し
        response = await client.put(
            f"{BACKEND_API_BASE}/api/avatars/stats",
            json=update_data,
            headers={"Authorization": "Bearer internal_ml_token"},  # 内部通信用トークン
        )

        if response.status_code != 200:
            print(f"ユーザー統計更新エラー: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"ユーザー統計更新エラー: {str(e)}")


def generate_contextual_feedback(message: str, context: Optional[Dict] = None) -> str:
    """文脈に応じたフィードバックを生成"""

    feedback_templates = {
        "question": [
            "素晴らしい質問ですね！その疑問を持つことは学習の第一歩です。",
            "良い質問です。一緒に考えてみましょう。",
            "その質問は多くの人が持つ疑問です。詳しく調べてみましょう。",
        ],
        "struggle": [
            "難しいと感じることは成長の証拠です。一歩ずつ進んでいきましょう。",
            "困った時こそ学習のチャンスです。諦めずに取り組みましょう。",
            "困難を乗り越えることで、より強くなれます。",
        ],
        "success": [
            "素晴らしいです！その調子で頑張りましょう。",
            "よくできました！成功体験を積み重ねていきましょう。",
            "完璧です！この成功を次の学習に活かしましょう。",
        ],
        "motivation": [
            "学習への意欲が素晴らしいです！その気持ちを大切にしましょう。",
            "やる気が感じられます。きっと良い結果が得られるでしょう。",
            "積極的な姿勢が素晴らしいです。継続は力なりです。",
        ],
    }

    # メッセージの内容からフィードバックタイプを判定
    message_lower = message.lower()

    if any(
        word in message_lower for word in ["わからない", "難しい", "困った", "できない"]
    ):
        feedback_type = "struggle"
    elif any(word in message_lower for word in ["できた", "成功", "完了", "解けた"]):
        feedback_type = "success"
    elif any(
        word in message_lower for word in ["頑張る", "挑戦", "学びたい", "知りたい"]
    ):
        feedback_type = "motivation"
    elif "?" in message or "ですか" in message or "でしょうか" in message:
        feedback_type = "question"
    else:
        feedback_type = "question"

    import random

    return random.choice(
        feedback_templates.get(feedback_type, feedback_templates["question"])
    )


def generate_suggestions(message: str, context: Optional[Dict] = None) -> List[str]:
    """メッセージに基づいて学習提案を生成"""

    suggestions = []
    message_lower = message.lower()

    # 数学関連の提案
    if any(word in message_lower for word in ["数学", "計算", "問題", "式"]):
        suggestions.extend(
            [
                "数学の基礎問題から始めてみましょう",
                "計算練習のクエストに挑戦してみませんか？",
                "図形や関数の問題も解いてみましょう",
            ]
        )

    # 理科関連の提案
    elif any(
        word in message_lower
        for word in ["理科", "科学", "実験", "物理", "化学", "生物"]
    ):
        suggestions.extend(
            [
                "理科の実験クエストに参加してみましょう",
                "身近な現象について調べてみませんか？",
                "科学の不思議を探求してみましょう",
            ]
        )

    # 国語関連の提案
    elif any(word in message_lower for word in ["国語", "読書", "文章", "作文"]):
        suggestions.extend(
            [
                "読書感想文を書いてみませんか？",
                "文章読解のクエストに挑戦してみましょう",
                "語彙力を高める学習をしてみましょう",
            ]
        )

    # 学習方法の提案
    else:
        suggestions.extend(
            [
                "今日の学習目標を設定してみましょう",
                "復習の時間を作ってみませんか？",
                "新しいクエストに挑戦してみましょう",
            ]
        )

    return suggestions[:3]  # 最大3つまで


@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "timestamp": datetime.now()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
