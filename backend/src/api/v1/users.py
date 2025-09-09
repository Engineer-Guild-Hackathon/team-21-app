from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.core.security import get_current_active_user
from src.domain.models.user import User
from src.domain.schemas.user import UserCreate, UserResponse, UserUpdate
from src.domain.types.user import UserId, UserProfile
from src.infrastructure.database import get_db
from src.infrastructure.repositories.user_repository import SQLAlchemyUserRepository
from src.services.user.user_service import UserService

router = APIRouter()


def get_user_service(db: Session = Depends(get_db)) -> UserService:
    """UserServiceの依存性注入"""
    repository = SQLAlchemyUserRepository(db)
    return UserService(repository)


@router.post("/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate, service: UserService = Depends(get_user_service)
):
    """新規ユーザーの作成"""
    try:
        user = await service.create_user(
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/me", response_model=UserResponse)
async def read_user_me(current_user: User = Depends(get_current_active_user)):
    """現在のユーザー情報を取得"""
    return current_user


@router.get("/{user_id}", response_model=UserResponse)
async def read_user(user_id: int, service: UserService = Depends(get_user_service)):
    """指定したユーザーの情報を取得"""
    user = await service.get_user(UserId(user_id))
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/", response_model=list[UserResponse])
async def read_users(
    skip: int = 0, limit: int = 100, service: UserService = Depends(get_user_service)
):
    """ユーザー一覧を取得"""
    return await service.list_users(skip=skip, limit=limit)


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    service: UserService = Depends(get_user_service),
    current_user: User = Depends(get_current_active_user),
):
    """ユーザー情報を更新"""
    # 自分自身または管理者のみ更新可能
    if user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # プロフィール情報の更新
    profile = UserProfile(
        full_name=user_data.full_name,
        avatar_url=user_data.avatar_url,
        bio=user_data.bio,
    )

    user = await service.update_profile(UserId(user_id), profile)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    service: UserService = Depends(get_user_service),
    current_user: User = Depends(get_current_active_user),
):
    """ユーザーを削除"""
    # 管理者のみ削除可能
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    if not await service.delete_user(UserId(user_id)):
        raise HTTPException(status_code=404, detail="User not found")

    return {"ok": True}
