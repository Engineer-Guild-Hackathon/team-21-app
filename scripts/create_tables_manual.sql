-- 手動でテーブルを作成するスクリプト

-- 1. usersテーブル
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR NOT NULL UNIQUE,
    hashed_password VARCHAR NOT NULL,
    full_name VARCHAR,
    role VARCHAR NOT NULL DEFAULT 'student',
    class_id INTEGER,
    is_active BOOLEAN NOT NULL DEFAULT true,
    is_verified BOOLEAN NOT NULL DEFAULT false,
    terms_accepted BOOLEAN NOT NULL DEFAULT false,
    terms_accepted_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2. classesテーブル（class_idカラムを含む）
CREATE TABLE IF NOT EXISTS classes (
    id SERIAL PRIMARY KEY,
    class_id VARCHAR NOT NULL UNIQUE,
    name VARCHAR NOT NULL,
    description TEXT,
    teacher_id INTEGER NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (teacher_id) REFERENCES users(id)
);

-- 3. learning_progressテーブル
CREATE TABLE IF NOT EXISTS learning_progress (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    quest_id INTEGER,
    progress_percentage INTEGER NOT NULL DEFAULT 0,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (class_id) REFERENCES classes(id)
);

-- 4. avatarsテーブル
CREATE TABLE IF NOT EXISTS avatars (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    description TEXT,
    image_url VARCHAR NOT NULL,
    category VARCHAR NOT NULL DEFAULT 'character',
    rarity VARCHAR NOT NULL DEFAULT 'common',
    is_active BOOLEAN NOT NULL DEFAULT true,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 5. user_avatarsテーブル
CREATE TABLE IF NOT EXISTS user_avatars (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    avatar_id INTEGER NOT NULL,
    is_current BOOLEAN NOT NULL DEFAULT false,
    obtained_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (avatar_id) REFERENCES avatars(id)
);

-- 6. user_statsテーブル
CREATE TABLE IF NOT EXISTS user_stats (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE,
    grit_level FLOAT NOT NULL DEFAULT 1.0,
    collaboration_level FLOAT NOT NULL DEFAULT 1.0,
    self_regulation_level FLOAT NOT NULL DEFAULT 1.0,
    emotional_intelligence_level FLOAT NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 7. questsテーブル
CREATE TYPE quest_type AS ENUM ('daily', 'weekly', 'monthly', 'special');
CREATE TYPE quest_difficulty AS ENUM ('easy', 'medium', 'hard');
CREATE TYPE quest_status AS ENUM ('draft', 'active', 'inactive', 'completed');

CREATE TABLE IF NOT EXISTS quests (
    id SERIAL PRIMARY KEY,
    title VARCHAR NOT NULL,
    description TEXT NOT NULL,
    quest_type quest_type NOT NULL DEFAULT 'daily',
    difficulty quest_difficulty NOT NULL DEFAULT 'easy',
    target_skills TEXT[] NOT NULL DEFAULT '{}',
    steps TEXT[] NOT NULL DEFAULT '{}',
    reward_points INTEGER NOT NULL DEFAULT 0,
    status quest_status NOT NULL DEFAULT 'draft',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 8. quest_progressesテーブル
CREATE TABLE IF NOT EXISTS quest_progresses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    quest_id INTEGER NOT NULL,
    progress_percentage INTEGER NOT NULL DEFAULT 0,
    status quest_status NOT NULL DEFAULT 'draft',
    completed_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (quest_id) REFERENCES quests(id)
);

-- 9. quest_rewardsテーブル
CREATE TABLE IF NOT EXISTS quest_rewards (
    id SERIAL PRIMARY KEY,
    quest_id INTEGER NOT NULL,
    reward_type VARCHAR NOT NULL,
    reward_value VARCHAR NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (quest_id) REFERENCES quests(id)
);

-- インデックス作成
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_classes_class_id ON classes(class_id);
CREATE INDEX IF NOT EXISTS idx_classes_teacher_id ON classes(teacher_id);
CREATE INDEX IF NOT EXISTS idx_learning_progress_user_id ON learning_progress(user_id);
CREATE INDEX IF NOT EXISTS idx_learning_progress_class_id ON learning_progress(class_id);
CREATE INDEX IF NOT EXISTS idx_quest_progresses_user_id ON quest_progresses(user_id);
CREATE INDEX IF NOT EXISTS idx_quest_progresses_quest_id ON quest_progresses(quest_id);
