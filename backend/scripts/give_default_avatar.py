#!/usr/bin/env python3
"""
既存ユーザーにデフォルトアバターを付与するスクリプト
"""

import asyncio
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from sqlalchemy import text
from src.infrastructure.database import get_db


async def give_default_avatar():
    """既存ユーザーにデフォルトアバターを付与"""

    async for db in get_db():
        try:
            # デフォルトアバター（ひよこ）を取得
            default_avatar_stmt = await db.execute(
                text("SELECT id FROM avatars WHERE name = 'ひよこ'")
            )
            default_avatar = default_avatar_stmt.fetchone()

            if not default_avatar:
                print("❌ デフォルトアバター（ひよこ）が見つかりません")
                return

            avatar_id = default_avatar[0]
            print(f"✅ デフォルトアバターID: {avatar_id}")

            # 全ユーザーを取得
            users_stmt = await db.execute(text("SELECT id, full_name FROM users"))
            users = users_stmt.fetchall()

            print(f"📊 対象ユーザー数: {len(users)}")

            for user in users:
                user_id = user[0]
                user_name = user[1] or "Unknown"

                # 既にアバターを持っているかチェック
                existing_stmt = await db.execute(
                    text("SELECT id FROM user_avatars WHERE user_id = :user_id"),
                    {"user_id": user_id},
                )
                existing = existing_stmt.fetchone()

                if existing:
                    print(f"⏭️  {user_name} (ID: {user_id}) - 既にアバターを所持")
                    continue

                # デフォルトアバターを付与
                await db.execute(
                    text("""INSERT INTO user_avatars (user_id, avatar_id, is_current, unlocked_at)
                           VALUES (:user_id, :avatar_id, true, NOW())"""),
                    {"user_id": user_id, "avatar_id": avatar_id},
                )

                # ユーザー統計を作成（存在しない場合）
                stats_stmt = await db.execute(
                    text("SELECT id FROM user_stats WHERE user_id = :user_id"),
                    {"user_id": user_id},
                )
                stats = stats_stmt.fetchone()

                if not stats:
                    await db.execute(
                        text("""INSERT INTO user_stats (user_id, total_avatars_unlocked, created_at, updated_at)
                               VALUES (:user_id, 1, NOW(), NOW())"""),
                        {"user_id": user_id},
                    )
                else:
                    await db.execute(
                        text(
                            "UPDATE user_stats SET total_avatars_unlocked = total_avatars_unlocked + 1 WHERE user_id = :user_id"
                        ),
                        {"user_id": user_id},
                    )

                print(f"✅ {user_name} (ID: {user_id}) - デフォルトアバターを付与")

            await db.commit()
            print("🎉 デフォルトアバターの付与が完了しました！")

        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")
            await db.rollback()
        finally:
            break


if __name__ == "__main__":
    asyncio.run(give_default_avatar())
