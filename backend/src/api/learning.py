from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_learning_status():
    return {"message": "学習状態を取得するエンドポイント（実装予定）"}
