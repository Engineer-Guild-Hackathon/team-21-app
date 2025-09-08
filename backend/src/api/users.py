from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.auth import get_current_user
from domain.models.user import User
from domain.schemas.auth import UserResponse
from infrastructure.database import get_db

router = APIRouter()


@router.get("/", response_model=List[UserResponse])
async def list_users(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> List[User]:
    users = db.query(User).all()
    return users


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
    return user
