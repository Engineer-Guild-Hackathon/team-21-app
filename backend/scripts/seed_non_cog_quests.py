"""
非認知能力向上クエスト（行動型テンプレート6件）のシードスクリプト。

実行例:
  docker-compose exec backend python scripts/seed_non_cog_quests.py
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

try:
    # 通常ケース（PYTHONPATHにプロジェクトルートが入っている）
    from src.domain.models.quest import Quest, QuestDifficulty, QuestType
    from src.infrastructure.database import SessionLocal
except ModuleNotFoundError:
    # フォールバック：実行環境でパス未設定の場合にのみ追加
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.domain.models.quest import Quest, QuestDifficulty, QuestType
    from src.infrastructure.database import SessionLocal

QUEST_DEFS = [
    {
        "title": "小さな助け合いレポート",
        "description": "友達/家族を1回手助けし、内容と相手の反応を記録しよう",
        "quest_type": QuestType.COLLABORATION,
        "difficulty": QuestDifficulty.EASY,
        "target_skill": "協働",
        "estimated_duration": 5,
        "quest_config": {"template": "helping_report"},
        "experience_points": 80,
        "coins": 20,
        "is_daily": False,
        "sort_order": 10,
    },
    {
        "title": "できたこと日記",
        "description": "今日の『できたこと』を3つ書き、難易度を星で付けよう",
        "quest_type": QuestType.DAILY_LOG,
        "difficulty": QuestDifficulty.EASY,
        "target_skill": "自信",
        "estimated_duration": 8,
        "quest_config": {"template": "achievement_diary"},
        "experience_points": 100,
        "coins": 30,
        "is_daily": True,
        "sort_order": 11,
    },
    {
        "title": "困ったら聞こうチャレンジ",
        "description": "質問文を1つ作り、丁寧に相手へ聞いて結果を記録しよう",
        "quest_type": QuestType.COLLABORATION,
        "difficulty": QuestDifficulty.MEDIUM,
        "target_skill": "協働/自信",
        "estimated_duration": 10,
        "quest_config": {"template": "ask_for_help"},
        "experience_points": 120,
        "coins": 40,
        "is_daily": False,
        "sort_order": 12,
    },
    {
        "title": "途中でやめないリレー（3日連続）",
        "description": "同じ短い習慣を3日連続で続けて記録しよう",
        "quest_type": QuestType.DAILY_LOG,
        "difficulty": QuestDifficulty.MEDIUM,
        "target_skill": "やり抜く力",
        "estimated_duration": 3,
        "quest_config": {"template": "streak_habit"},
        "experience_points": 150,
        "coins": 60,
        "is_daily": False,
        "sort_order": 13,
    },
    {
        "title": "ミニ先生",
        "description": "だれかに1つだけ教えて、説明→質問→確認を記録しよう",
        "quest_type": QuestType.COLLABORATION,
        "difficulty": QuestDifficulty.MEDIUM,
        "target_skill": "協働/自信",
        "estimated_duration": 12,
        "quest_config": {"template": "mini_teacher"},
        "experience_points": 140,
        "coins": 50,
        "is_daily": False,
        "sort_order": 14,
    },
    {
        "title": "みんな違ってみんないい",
        "description": "自分と違う意見の良いところを2点書いてみよう",
        "quest_type": QuestType.STORY_CREATION,
        "difficulty": QuestDifficulty.EASY,
        "target_skill": "情動/協働",
        "estimated_duration": 7,
        "quest_config": {"template": "respect_different_opinion"},
        "experience_points": 90,
        "coins": 25,
        "is_daily": False,
        "sort_order": 15,
    },
]


async def upsert_quests():
    async with SessionLocal() as session:
        for q in QUEST_DEFS:
            # 既存チェック（タイトルで）
            existing = await session.execute(
                select(Quest).where(Quest.title == q["title"])
            )
            row = existing.scalar_one_or_none()

            if row:
                # 既存は更新（説明や設定の追従）
                row.description = q["description"]
                row.quest_type = q["quest_type"]
                row.difficulty = q["difficulty"]
                row.target_skill = q["target_skill"]
                row.estimated_duration = q["estimated_duration"]
                row.quest_config = q["quest_config"]
                row.experience_points = q["experience_points"]
                row.coins = q["coins"]
                row.is_daily = q["is_daily"]
                row.sort_order = q["sort_order"]
            else:
                session.add(
                    Quest(
                        title=q["title"],
                        description=q["description"],
                        quest_type=q["quest_type"],
                        difficulty=q["difficulty"],
                        target_skill=q["target_skill"],
                        estimated_duration=q["estimated_duration"],
                        quest_config=q["quest_config"],
                        experience_points=q["experience_points"],
                        coins=q["coins"],
                        is_daily=q["is_daily"],
                        sort_order=q["sort_order"],
                        required_level=1,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                )

        await session.commit()


def main():
    asyncio.run(upsert_quests())


if __name__ == "__main__":
    main()
