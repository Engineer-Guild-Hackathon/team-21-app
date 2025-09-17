#!/bin/bash
# データベース初期化スクリプト
# このスクリプトはデータベースとテーブルが存在しない場合に作成します

set -e

echo "=== データベース初期化スクリプト ==="

# 環境変数の確認
if [ -z "$DATABASE_URL" ]; then
    echo "エラー: DATABASE_URLが設定されていません"
    exit 1
fi

echo "DATABASE_URL: $DATABASE_URL"

# データベース接続テスト
echo "データベース接続をテスト中..."
psql "$DATABASE_URL" -c "SELECT 1;" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "エラー: データベースに接続できません"
    exit 1
fi

echo "データベース接続成功"

# テーブル存在確認と作成
echo "テーブルの存在確認と作成中..."

# usersテーブル
psql "$DATABASE_URL" -c "
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'student',
    class_id VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    terms_accepted BOOLEAN DEFAULT false,
    terms_accepted_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);"

# classesテーブル
psql "$DATABASE_URL" -c "
CREATE TABLE IF NOT EXISTS classes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    teacher_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);"

# learning_progressテーブル
psql "$DATABASE_URL" -c "
CREATE TABLE IF NOT EXISTS learning_progress (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    class_id INTEGER REFERENCES classes(id),
    skill_type VARCHAR(100) NOT NULL,
    current_level DECIMAL(5,2) DEFAULT 1.0,
    progress_percentage DECIMAL(5,2) DEFAULT 0.0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);"

# クエスト関連テーブル
psql "$DATABASE_URL" -c "
CREATE TYPE questtype AS ENUM ('DAILY_LOG', 'PLANT_CARE', 'STORY_CREATION', 'COLLABORATION', 'EMOTION_REGULATION', 'PROBLEM_SOLVING');
CREATE TYPE questdifficulty AS ENUM ('EASY', 'MEDIUM', 'HARD');
CREATE TYPE queststatus AS ENUM ('NOT_STARTED', 'IN_PROGRESS', 'COMPLETED');

CREATE TABLE IF NOT EXISTS quests (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    quest_type questtype NOT NULL,
    difficulty questdifficulty NOT NULL,
    target_skill VARCHAR(100),
    experience_points INTEGER DEFAULT 0,
    coins INTEGER DEFAULT 0,
    badge_id INTEGER,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS quest_progresses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    quest_id INTEGER REFERENCES quests(id),
    status queststatus DEFAULT 'NOT_STARTED',
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 1,
    progress_percentage DECIMAL(5,2) DEFAULT 0.0,
    started_date TIMESTAMP,
    completed_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, quest_id)
);

CREATE TABLE IF NOT EXISTS quest_rewards (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    quest_id INTEGER REFERENCES quests(id),
    reward_type VARCHAR(50) NOT NULL,
    reward_value INTEGER NOT NULL,
    reward_data JSONB,
    is_claimed BOOLEAN DEFAULT false,
    claimed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"

# アバター関連テーブル
psql "$DATABASE_URL" -c "
CREATE TABLE IF NOT EXISTS avatars (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    image_url VARCHAR(500),
    category VARCHAR(100),
    rarity VARCHAR(50),
    is_active BOOLEAN DEFAULT true,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_avatars (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    avatar_id INTEGER REFERENCES avatars(id),
    is_current BOOLEAN DEFAULT false,
    acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, avatar_id)
);

CREATE TABLE IF NOT EXISTS user_stats (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    grit_level DECIMAL(5,2) DEFAULT 1.0,
    collaboration_level DECIMAL(5,2) DEFAULT 1.0,
    self_regulation_level DECIMAL(5,2) DEFAULT 1.0,
    emotional_intelligence_level DECIMAL(5,2) DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id)
);
"

echo "=== データベース初期化完了 ==="
echo "作成されたテーブル:"
psql "$DATABASE_URL" -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;"
