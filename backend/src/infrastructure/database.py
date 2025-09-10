import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


# データベース設定 - 環境変数から動的に読み取り
def get_database_url():
    """環境に応じてデータベースURLを取得"""
    # 環境変数が設定されている場合はそれを使用
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")

    # 環境変数が設定されていない場合、環境に応じてデフォルトを決定
    environment = os.getenv("ENVIRONMENT", "development")

    if environment == "production":
        # 本番環境用のデフォルト（通常は環境変数で設定される）
        return "postgresql+asyncpg://postgres:postgres@db:5432/noncog"
    else:
        # 開発環境用のデフォルト（ローカルPostgreSQL）
        return "postgresql+asyncpg://postgres:postgres@localhost:5432/noncog"


SQLALCHEMY_DATABASE_URL = get_database_url()

engine = create_async_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """データベースセッションを取得"""
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
