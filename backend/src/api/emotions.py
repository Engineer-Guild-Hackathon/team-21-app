from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_emotions() -> dict[str, str]:
    """感情一覧を取得"""
    return {"emotions": "実装予定"}