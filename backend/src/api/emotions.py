from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_emotion_status():
    return {"message": "感情分析状態を取得するエンドポイント（実装予定）"}
