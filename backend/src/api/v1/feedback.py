from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_feedback() -> dict[str, str]:
    """フィードバックを取得"""
    return {"feedback": "実装予定"}
