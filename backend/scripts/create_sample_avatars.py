#!/usr/bin/env python3
"""
サンプルのアバター・称号データを作成するスクリプト
"""

import asyncio
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.domain.models.avatar import Avatar, AvatarPart, Title
from src.infrastructure.database import get_db


async def create_sample_avatars():
    """サンプルのアバター・称号データを作成"""

    async for db in get_db():
        try:
            # サンプルアバター
            avatars = [
                # 基本アバター
                Avatar(
                    name="ひよこ",
                    description="可愛いひよこアバター。初心者におすすめ！",
                    image_url="/images/avatars/chick.png",
                    category="animal",
                    rarity="common",
                    unlock_condition_type=None,
                    unlock_condition_value=None,
                    sort_order=1,
                ),
                Avatar(
                    name="うさぎ",
                    description="ぴょんぴょん跳ねる元気なうさぎ",
                    image_url="/images/avatars/rabbit.png",
                    category="animal",
                    rarity="common",
                    unlock_condition_type="quest_count",
                    unlock_condition_value=5,
                    sort_order=2,
                ),
                Avatar(
                    name="パンダ",
                    description="愛らしいパンダアバター",
                    image_url="/images/avatars/panda.png",
                    category="animal",
                    rarity="rare",
                    unlock_condition_type="quest_count",
                    unlock_condition_value=15,
                    sort_order=3,
                ),
                Avatar(
                    name="ロボット",
                    description="未来感あふれるロボットアバター",
                    image_url="/images/avatars/robot.png",
                    category="robot",
                    rarity="rare",
                    unlock_condition_type="level_reach",
                    unlock_condition_value=3,
                    sort_order=4,
                ),
                Avatar(
                    name="ドラゴン",
                    description="伝説のドラゴンアバター",
                    image_url="/images/avatars/dragon.png",
                    category="mythical",
                    rarity="legendary",
                    unlock_condition_type="quest_count",
                    unlock_condition_value=50,
                    sort_order=5,
                ),
            ]

            # サンプルアバターパーツ
            avatar_parts = [
                AvatarPart(
                    name="魔法の帽子",
                    description="キラキラ光る魔法の帽子",
                    image_url="/images/avatar_parts/magic_hat.png",
                    part_type="hat",
                    rarity="rare",
                    unlock_condition_type="quest_count",
                    unlock_condition_value=10,
                    sort_order=1,
                ),
                AvatarPart(
                    name="サングラス",
                    description="カッコいいサングラス",
                    image_url="/images/avatar_parts/sunglasses.png",
                    part_type="glasses",
                    rarity="common",
                    unlock_condition_type="quest_count",
                    unlock_condition_value=3,
                    sort_order=2,
                ),
                AvatarPart(
                    name="王冠",
                    description="高貴な王冠",
                    image_url="/images/avatar_parts/crown.png",
                    part_type="hat",
                    rarity="epic",
                    unlock_condition_type="quest_count",
                    unlock_condition_value=30,
                    sort_order=3,
                ),
            ]

            # サンプル称号
            titles = [
                Title(
                    name="学習の始まり",
                    description="初めてのクエストを完了しました",
                    icon_url="/images/titles/learning_start.png",
                    category="learning",
                    rarity="common",
                    unlock_condition_type="quest_count",
                    unlock_condition_value=1,
                    unlock_condition_description="1つのクエストを完了する",
                    sort_order=1,
                ),
                Title(
                    name="継続学習マスター",
                    description="7日連続で学習を継続",
                    icon_url="/images/titles/streak_master.png",
                    category="learning",
                    rarity="rare",
                    unlock_condition_type="streak_days",
                    unlock_condition_value=7,
                    unlock_condition_description="7日連続で学習する",
                    sort_order=2,
                ),
                Title(
                    name="クエストクリア王",
                    description="たくさんのクエストを完了",
                    icon_url="/images/titles/quest_king.png",
                    category="quest",
                    rarity="epic",
                    unlock_condition_type="quest_count",
                    unlock_condition_value=25,
                    unlock_condition_description="25個のクエストを完了する",
                    sort_order=3,
                ),
                Title(
                    name="協力の達人",
                    description="チームワークを大切にする学習者",
                    icon_url="/images/titles/cooperation_master.png",
                    category="cooperation",
                    rarity="rare",
                    unlock_condition_type="skill_level",
                    unlock_condition_value=3.0,
                    unlock_condition_description="協力スキルをレベル3まで上げる",
                    sort_order=4,
                ),
                Title(
                    name="非認知能力の達人",
                    description="すべてのスキルをバランスよく向上",
                    icon_url="/images/titles/skill_master.png",
                    category="special",
                    rarity="legendary",
                    unlock_condition_type="skill_level",
                    unlock_condition_value=4.0,
                    unlock_condition_description="全スキルの平均レベルを4.0まで上げる",
                    sort_order=5,
                ),
                Title(
                    name="感情のマスター",
                    description="感情を理解し、コントロールできる",
                    icon_url="/images/titles/emotion_master.png",
                    category="learning",
                    rarity="epic",
                    unlock_condition_type="skill_level",
                    unlock_condition_value=4.5,
                    unlock_condition_description="感情知能スキルをレベル4.5まで上げる",
                    sort_order=6,
                ),
            ]

            # データベースに保存
            for avatar in avatars:
                db.add(avatar)

            for part in avatar_parts:
                db.add(part)

            for title in titles:
                db.add(title)

            await db.commit()
            print("✅ サンプルのアバター・称号データを作成しました")
            print(f"   - アバター: {len(avatars)}個")
            print(f"   - アバターパーツ: {len(avatar_parts)}個")
            print(f"   - 称号: {len(titles)}個")

        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")
            await db.rollback()
        finally:
            break


if __name__ == "__main__":
    asyncio.run(create_sample_avatars())
