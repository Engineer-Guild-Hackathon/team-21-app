#!/bin/bash

# ローカル開発環境セットアップスクリプト
# 使用方法: ./scripts/setup-local-dev.sh

set -e

# 色付きログ関数
log_info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

log_info "🚀 ローカル開発環境セットアップ開始"

# 前提条件チェック
check_prerequisites() {
    log_info "前提条件をチェック中..."
    
    if ! command -v docker &> /dev/null; then
        log_error "docker CLI がインストールされていません"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose CLI がインストールされていません"
        exit 1
    fi
    
    log_info "✅ 前提条件チェック完了"
}

# 環境変数ファイルの作成
create_env_files() {
    log_info "環境変数ファイルを作成中..."
    
    # バックエンド用の環境変数ファイル
    if [ ! -f "backend/.env.local" ]; then
        cp backend/env.local.example backend/.env.local
        log_info "✅ backend/.env.local を作成しました"
    else
        log_info "✅ backend/.env.local は既に存在します"
    fi
    
    # フロントエンド用の環境変数ファイル
    if [ ! -f "frontend/.env.local" ]; then
        cp frontend/env.local.example frontend/.env.local
        log_info "✅ frontend/.env.local を作成しました"
    else
        log_info "✅ frontend/.env.local は既に存在します"
    fi
    
    log_info "✅ 環境変数ファイル作成完了"
}

# Docker Compose でサービス起動
start_services() {
    log_info "Docker Compose でサービスを起動中..."
    
    # 既存のコンテナを停止・削除
    docker-compose down -v
    
    # サービスを起動
    docker-compose up -d db redis
    
    # データベースが起動するまで待機
    log_info "データベースの起動を待機中..."
    sleep 10
    
    log_info "✅ サービス起動完了"
}

# データベースのセットアップ
setup_database() {
    log_info "データベースをセットアップ中..."
    
    # バックエンドコンテナを起動（データベースセットアップ用）
    docker-compose up -d backend
    
    # バックエンドが起動するまで待機
    log_info "バックエンドの起動を待機中..."
    sleep 15
    
    # マイグレーション実行
    log_info "データベースマイグレーションを実行中..."
    docker-compose exec backend alembic upgrade head
    
    # シードデータ実行
    log_info "シードデータを実行中..."
    docker-compose exec backend python scripts/seed_data.py
    
    log_info "✅ データベースセットアップ完了"
}

# フロントエンドの起動
start_frontend() {
    log_info "フロントエンドを起動中..."
    
    docker-compose up -d frontend
    
    log_info "✅ フロントエンド起動完了"
}

# セットアップ完了の確認
verify_setup() {
    log_info "セットアップ完了を確認中..."
    
    # サービス状態確認
    docker-compose ps
    
    log_info "✅ セットアップ完了確認"
}

# アクセス情報の表示
show_access_info() {
    log_info "🎉 ローカル開発環境セットアップ完了！"
    log_info ""
    log_info "📱 アクセス情報:"
    log_info "  フロントエンド: http://localhost:3000"
    log_info "  バックエンドAPI: http://localhost:8000"
    log_info "  API ドキュメント: http://localhost:8000/docs"
    log_info ""
    log_info "🔑 デモアカウント:"
    log_info "  生徒: student@example.com / password123"
    log_info "  保護者: parent@example.com / password123"
    log_info "  教師: teacher@example.com / password123"
    log_info ""
    log_info "🛠️ 開発コマンド:"
    log_info "  全サービス起動: docker-compose up"
    log_info "  全サービス停止: docker-compose down"
    log_info "  ログ確認: docker-compose logs -f [service-name]"
    log_info "  データベースリセット: docker-compose down -v && docker-compose up -d"
}

# メイン実行
main() {
    check_prerequisites
    create_env_files
    start_services
    setup_database
    start_frontend
    verify_setup
    show_access_info
}

# スクリプト実行
main "$@"
