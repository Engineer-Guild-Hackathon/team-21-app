# データベース初期データ投入ガイド

## 概要

このガイドでは、Non-Cog Learning Platform のデータベースに初期データを投入する方法について説明します。

## 初期データの構造

```python
# backend/src/infrastructure/seed_data.py

from datetime import datetime
from sqlalchemy.orm import Session
from ..domain.models.user import User
from .database import SessionLocal
from ..core.security import get_password_hash

def create_demo_users(db: Session) -> None:
    """デモユーザーの作成"""
    demo_users = [
        {
            "email": "student@demo.com",
            "password": "student123",
            "full_name": "デモ生徒",
            "role": "student",
            "is_active": True
        },
        {
            "email": "teacher@demo.com",
            "password": "teacher123",
            "full_name": "デモ教師",
            "role": "teacher",
            "is_active": True
        },
        {
            "email": "parent@demo.com",
            "password": "parent123",
            "full_name": "デモ保護者",
            "role": "parent",
            "is_active": True
        }
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
                created_at=datetime.utcnow()
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
```

## 実行方法

1. スクリプトの配置

   - 上記のコードを`backend/src/infrastructure/seed_data.py`として保存

2. 実行コマンド

   ```bash
   # バックエンドコンテナ内で実行
   python -m src.infrastructure.seed_data
   ```

3. Makefile に追加するコマンド
   ```makefile
   db-seed: ## データベースに初期データを投入
       @echo "$(CYAN)初期データを投入しています...$(RESET)"
       @docker-compose exec backend python -m src.infrastructure.seed_data
       @echo "$(GREEN)初期データの投入が完了しました$(RESET)"
   ```

## デモアカウント

以下のアカウントでログインできます：

1. 生徒アカウント

   - Email: student@demo.com
   - Password: student123

2. 教師アカウント

   - Email: teacher@demo.com
   - Password: teacher123

3. 保護者アカウント
   - Email: parent@demo.com
   - Password: parent123

## 注意事項

- 本番環境では、デモアカウントのパスワードを必ず変更してください
- 初期データは開発環境でのみ使用することを推奨します
- パスワードは環境変数で管理することを推奨します

## トラブルシューティング

1. データベース接続エラー

   - PostgreSQL サービスが起動していることを確認
   - データベース接続情報が正しいことを確認

2. 重複エラー
   - スクリプトは既存のユーザーをスキップするように設計されています
   - エラーが発生した場合は、既存データを確認してください

## 開発フロー

1. データベースのマイグレーション実行

   ```bash
   make db-migrate
   ```

2. 初期データの投入

   ```bash
   make db-seed
   ```

3. 動作確認
   - 各デモアカウントでログインを試行
   - 各ロールの機能が正しく動作することを確認
