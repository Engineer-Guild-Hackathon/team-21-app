#!/usr/bin/env python3
"""
本番用デモアカウント作成スクリプト
"""

import asyncio
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.security import get_password_hash
from src.domain.models.classroom import Class, LearningProgress
from src.domain.models.user import User
from src.infrastructure.database import get_db


async def create_demo_accounts():
    """デモアカウントを作成"""
    print("🚀 デモアカウント作成を開始します...")
    async for db in get_db():
        try:
            # 1. 教師アカウントを作成
            teacher = User(
                email="teacher@noncog.com",
                hashed_password=get_password_hash("teacher123"),
                full_name="田中先生",
                role="teacher",
            )
            db.add(teacher)
            await db.flush()  # IDを取得するためにflush

            # 2. クラスを作成
            class_obj = Class(
                class_id="NON-001",
                name="5年1組",
                description="非認知能力学習クラス",
                teacher_id=teacher.id,
            )
            db.add(class_obj)
            await db.flush()

            # 3. 生徒アカウントを作成
            students_data = [
                {"name": "山田太郎", "email": "yamada@noncog.com"},
                {"name": "佐藤花子", "email": "sato@noncog.com"},
                {"name": "鈴木一郎", "email": "suzuki@noncog.com"},
                {"name": "高橋美咲", "email": "takahashi@noncog.com"},
                {"name": "伊藤健太", "email": "ito@noncog.com"},
            ]

            students = []
            for student_data in students_data:
                student = User(
                    email=student_data["email"],
                    hashed_password=get_password_hash("student123"),
                    full_name=student_data["name"],
                    role="student",
                    class_id=class_obj.id,
                )
                db.add(student)
                students.append(student)

            await db.flush()

            # 4. 保護者アカウントを作成
            parents_data = [
                {"name": "山田花子", "email": "yamada.parent@noncog.com"},
                {"name": "佐藤太郎", "email": "sato.parent@noncog.com"},
                {"name": "鈴木美香", "email": "suzuki.parent@noncog.com"},
                {"name": "高橋健一", "email": "takahashi.parent@noncog.com"},
                {"name": "伊藤由美", "email": "ito.parent@noncog.com"},
            ]

            for parent_data in parents_data:
                parent = User(
                    email=parent_data["email"],
                    hashed_password=get_password_hash("parent123"),
                    full_name=parent_data["name"],
                    role="parent",
                )
                db.add(parent)

            # 5. 学習進捗レコードを作成
            for student in students:
                progress = LearningProgress(
                    student_id=student.id,
                    class_id=class_obj.id,
                    grit_score=75.0,  # やり抜く力
                    collaboration_score=80.0,  # 協調性
                    self_regulation_score=70.0,  # 自己調整
                    emotional_intelligence_score=85.0,  # 感情知性
                    quests_completed=5,
                    total_learning_time=120,  # 2時間
                    retry_count=3,
                )
                db.add(progress)

            await db.commit()
            print("✅ デモアカウントが正常に作成されました！")
            print("\n📋 作成されたアカウント情報:")
            print(f"🏫 クラス: {class_obj.name} (ID: {class_obj.class_id})")
            print(
                f"👨‍🏫 教師: {teacher.full_name} ({teacher.email}) / パスワード: teacher123"
            )
            print("\n👨‍🎓 生徒アカウント:")
            for student in students:
                print(
                    f"  - {student.full_name} ({student.email}) / パスワード: student123"
                )
            print("\n👨‍👩‍👧‍👦 保護者アカウント:")
            for parent_data in parents_data:
                print(
                    f"  - {parent_data['name']} ({parent_data['email']}) / パスワード: parent123"
                )

        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")
            await db.rollback()
        finally:
            await db.close()


if __name__ == "__main__":
    asyncio.run(create_demo_accounts())
