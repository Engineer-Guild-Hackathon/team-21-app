from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_feedback():
    return {"message": "フィードバックを取得するエンドポイント（実装予定）"}
