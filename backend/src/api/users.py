from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..core.security import get_current_active_user
from ..domain.models.user import User
from ..domain.schemas.auth import UserResponse
from ..infrastructure.database import get_db

router = APIRouter()

@router.get("/", response_model=List[UserResponse])
async def read_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> List[User]:
    """ユーザー一覧を取得"""
    users = db.query(User).offset(skip).limit(limit).all()
    return users