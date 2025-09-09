from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ...core.security import decode_token
from ...domain.models.user import User
from ..database import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> User:
    """現在のユーザーを取得するミドルウェア

    トークンを検証し、対応するユーザーを返す。
    認証に失敗した場合は401エラーを発生させる。
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="認証情報が無効です",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        email = decode_token(token)
        if email is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """アクティブなユーザーを取得するミドルウェア

    現在のユーザーがアクティブかどうかを確認する。
    非アクティブの場合は400エラーを発生させる。
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="非アクティブユーザーです"
        )
    return current_user
