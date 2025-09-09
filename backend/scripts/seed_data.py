"""初期データ投入スクリプト"""

import sys
from pathlib import Path

# srcディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from src.core.security import get_password_hash
from src.domain.models.user import User
from src.infrastructure.database import SessionLocal

# デモユーザーデータ
DEMO_USERS = [
    {
        "email": "student@example.com",
        "password": "password123",
        "full_name": "山田太郎",
        "role": "student",
        "is_active": True,
        "is_verified": True,
    },
    {
        "email": "parent@example.com",
        "password": "password123",
        "full_name": "山田花子",
        "role": "parent",
        "is_active": True,
        "is_verified": True,
    },
    {
        "email": "teacher@example.com",
        "password": "password123",
        "full_name": "佐藤先生",
        "role": "teacher",
        "is_active": True,
        "is_verified": True,
    },
]


def create_demo_users(db: Session) -> None:
    """デモユーザーの作成"""
    for user_data in DEMO_USERS:
        # 既存ユーザーチェック
        existing_user = db.query(User).filter(User.email == user_data["email"]).first()
        if existing_user:
            print(f"ユーザー {user_data['email']} は既に存在します")
            continue

        # パスワードのハッシュ化
        hashed_password = get_password_hash(user_data["password"])

        # ユーザー作成
        user = User(
            email=user_data["email"],
            hashed_password=hashed_password,
            full_name=user_data["full_name"],
            role=user_data["role"],
            is_active=user_data["is_active"],
            is_verified=user_data["is_verified"],
        )
        db.add(user)
        print(f"ユーザー {user_data['email']} を作成しました")

    db.commit()


def main() -> None:
    """メイン実行関数"""
    db = SessionLocal()
    try:
        create_demo_users(db)
        print("デモユーザーの作成が完了しました")
    finally:
        db.close()


if __name__ == "__main__":
    main()
