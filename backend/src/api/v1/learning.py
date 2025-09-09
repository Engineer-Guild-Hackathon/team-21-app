from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_learning_status() -> dict[str, str]:
    """学習状態を取得"""
    return {"status": "実装予定"}
