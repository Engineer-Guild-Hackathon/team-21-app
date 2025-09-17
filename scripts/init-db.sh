#!/bin/bash
set -e

echo "=== データベース初期化スクリプト ==="

# 環境変数の確認
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL is not set"
    exit 1
fi

echo "DATABASE_URL: $DATABASE_URL"

# データベース接続テスト（最大30回、5秒間隔）
echo "データベース接続を確認中..."
for i in {1..30}; do
    if PGPASSWORD=postgres psql -h 127.0.0.1 -p 5432 -U postgres -d postgres -c "SELECT 1;" >/dev/null 2>&1; then
        echo "データベース接続成功"
        break
    fi
    echo "接続試行 $i/30..."
    sleep 5
done

# noncogデータベースの存在確認と作成
echo "noncogデータベースの確認..."
if ! PGPASSWORD=postgres psql -h 127.0.0.1 -p 5432 -U postgres -d noncog -c "SELECT 1;" >/dev/null 2>&1; then
    echo "noncogデータベースが存在しません。作成します..."
    PGPASSWORD=postgres psql -h 127.0.0.1 -p 5432 -U postgres -d postgres -c "CREATE DATABASE noncog;"
    echo "noncogデータベースを作成しました"
else
    echo "noncogデータベースは既に存在します"
fi

# Alembicマイグレーション実行
echo "Alembicマイグレーションを実行中..."
alembic upgrade head

echo "=== データベース初期化完了 ==="
