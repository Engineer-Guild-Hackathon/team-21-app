from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..core.security import get_current_active_user
from ..domain.models.emotion import EmotionRecord
from ..domain.models.user import User
from ..domain.schemas.chat import ChatRequest, ChatResponse
from ..infrastructure.database import get_db
from ..ml.dialogue.bert_dialogue import DialogueSystem
from ..ml.emotion_analysis.emotion_analyzer import EmotionAnalyzer
from ..ml.reinforcement.dqn_agent import DQNAgent

router = APIRouter()

# MLモデルのインスタンスを作成
dialogue_system = DialogueSystem()
emotion_analyzer = EmotionAnalyzer()
dqn_agent = DQNAgent()


@router.post("/respond", response_model=ChatResponse)
async def get_ai_response(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """AIチャットの応答を生成"""
    try:
        # 1. 感情分析
        emotion_result = emotion_analyzer.analyze_text(request.message)

        # 2. 行動選択（DQN）
        state = {
            "emotion": emotion_result["emotion"],
            "intensity": emotion_result["intensity"],
            "context": request.context if request.context else "general",
        }
        action = dqn_agent.select_action(state)

        # 3. 応答生成（BERT）
        response = dialogue_system.generate_response(
            user_message=request.message,
            emotion=emotion_result["emotion"],
            action=action,
        )

        # 4. 結果を保存
        emotion_record = EmotionRecord(
            user_id=current_user.id,
            emotion_type=emotion_result["emotion"],
            intensity=emotion_result["intensity"],
            context="chat",
        )
        db.add(emotion_record)
        db.commit()

        return {
            "message": response["text"],
            "emotion": {
                "type": response["emotion"],
                "intensity": response["intensity"],
            },
            "action": action,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"AI応答の生成中にエラーが発生しました: {str(e)}"
        )


@router.get("/history", response_model=List[ChatResponse])
async def get_chat_history(
    limit: Optional[int] = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    """チャット履歴を取得"""
    # チャット履歴の取得ロジックを実装
    return []
