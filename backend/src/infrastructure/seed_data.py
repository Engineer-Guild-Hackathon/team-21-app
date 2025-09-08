from datetime import datetime

from sqlalchemy.orm import Session

from ..core.security import get_password_hash
from ..domain.models.user import User
from .database import SessionLocal


def create_demo_users(db: Session) -> None:
    """デモユーザーの作成"""
    demo_users = [
        {
            "email": "student@demo.com",
            "password": "student123",
            "full_name": "デモ生徒",
            "role": "student",
            "is_active": True,
        },
        {
            "email": "teacher@demo.com",
            "password": "teacher123",
            "full_name": "デモ教師",
            "role": "teacher",
            "is_active": True,
        },
        {
            "email": "parent@demo.com",
            "password": "parent123",
            "full_name": "デモ保護者",
            "role": "parent",
            "is_active": True,
        },
    ]

    for user_data in demo_users:
        # ユーザーが存在しない場合のみ作成
        existing_user = db.query(User).filter(User.email == user_data["email"]).first()
        if not existing_user:
            db_user = User(
                email=user_data["email"],
                hashed_password=get_password_hash(user_data["password"]),
                full_name=user_data["full_name"],
                role=user_data["role"],
                is_active=user_data["is_active"],
                created_at=datetime.utcnow(),
            )
            db.add(db_user)

    db.commit()


def init_db() -> None:
    """データベースの初期化とシードデータの投入"""
    db = SessionLocal()
    try:
        create_demo_users(db)
        # 他の初期データ投入関数をここに追加
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
