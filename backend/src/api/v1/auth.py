from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.security import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_password_hash,
)
from src.domain.models.avatar import Avatar, UserAvatar, UserStats
from src.domain.models.user import User
from src.domain.schemas.auth import Token, UserCreate, UserResponse
from src.infrastructure.database import get_db

router = APIRouter()

ACCESS_TOKEN_EXPIRE_MINUTES = 30


async def create_default_user_setup(user_id: int, db: AsyncSession):
    """新規ユーザーにデフォルトアバターと統計情報を作成"""
    from sqlalchemy import select

    try:
        # 1. デフォルト統計情報を作成
        user_stats = UserStats(
            user_id=user_id,
            grit_level=1.0,
            collaboration_level=1.0,
            self_regulation_level=1.0,
            emotional_intelligence_level=1.0,
        )
        db.add(user_stats)

        # 2. デフォルトアバターを取得して設定（最初に見つかったアバターを使用）
        default_avatar_stmt = select(Avatar).limit(1)
        default_avatar_result = await db.execute(default_avatar_stmt)
        default_avatar = default_avatar_result.scalar_one_or_none()

        if default_avatar:
            user_avatar = UserAvatar(
                user_id=user_id,
                avatar_id=default_avatar.id,
                is_current=True,
            )
            db.add(user_avatar)

        await db.commit()
    except Exception as e:
        await db.rollback()
        print(f"Default user setup error: {e}")
        raise


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_db),
) -> Token:
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="メールアドレスまたはパスワードが正しくありません",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user.id, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate, db: AsyncSession = Depends(get_db)
) -> UserResponse:
    # メールアドレスの重複チェック
    from sqlalchemy import select

    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="このメールアドレスは既に登録されています",
        )

    # 利用規約への同意を確認
    if not user_data.terms_accepted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="利用規約への同意が必要です",
        )

    # パスワードのハッシュ化
    hashed_password = get_password_hash(user_data.password)

    # クラスIDの検証（生徒の場合）
    class_id = None
    if user_data.role == "student" and user_data.class_id:
        # クラスIDが存在するかチェック
        from src.domain.models.classroom import Class as ClassModel

        result = await db.execute(
            select(ClassModel).where(ClassModel.class_id == user_data.class_id)
        )
        class_obj = result.scalar_one_or_none()
        if class_obj:
            class_id = class_obj.id
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="指定されたクラスIDが見つかりません",
            )

    # ユーザーの作成
    db_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        role=user_data.role,
        class_id=class_id,
        terms_accepted=user_data.terms_accepted,
        terms_accepted_at=datetime.utcnow() if user_data.terms_accepted else None,
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)

    # 新規ユーザーにデフォルトアバターと統計情報を作成（エラーが発生しても登録は続行）
    # 一時的に無効化して基本的な登録機能をテスト
    # try:
    #     await create_default_user_setup(db_user.id, db)
    # except Exception as e:
    #     print(f"Warning: Default user setup failed: {e}")
    #     # デフォルトセットアップが失敗してもユーザー登録は成功とする

    return UserResponse(
        id=db_user.id,
        email=db_user.email,
        full_name=db_user.full_name,
        role=db_user.role,
        is_active=db_user.is_active,
        is_verified=db_user.is_verified,
        created_at=db_user.created_at,
        updated_at=db_user.updated_at,
    )


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> UserResponse:
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
    )
