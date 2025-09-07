.PHONY: setup start stop restart clean test lint logs help

# デフォルトのターゲット
.DEFAULT_GOAL := help

# 色の定義
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# 環境セットアップ
setup: ## 開発環境のセットアップ
	@echo "$(CYAN)環境をセットアップしています...$(RESET)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN).envファイルを作成しました$(RESET)"; \
	fi
	@docker-compose build
	@echo "$(GREEN)セットアップが完了しました$(RESET)"

# サービスの制御
start: ## 全サービスを起動
	@echo "$(CYAN)サービスを起動しています...$(RESET)"
	@docker-compose up -d
	@echo "$(GREEN)サービスが起動しました$(RESET)"
	@echo "フロントエンド: http://localhost:3000"
	@echo "バックエンドAPI: http://localhost:8000"
	@echo "MLサービス: http://localhost:8001"
	@echo "MLflow: http://localhost:5000"
	@echo "Spark UI: http://localhost:8080"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3001"

stop: ## 全サービスを停止
	@echo "$(CYAN)サービスを停止しています...$(RESET)"
	@docker-compose down
	@echo "$(GREEN)サービスを停止しました$(RESET)"

restart: stop start ## サービスを再起動

# 開発用コマンド
dev-frontend: ## フロントエンド開発サーバーを起動
	@echo "$(CYAN)フロントエンド開発サーバーを起動しています...$(RESET)"
	@docker-compose exec frontend npm run dev

dev-backend: ## バックエンド開発サーバーを起動
	@echo "$(CYAN)バックエンド開発サーバーを起動しています...$(RESET)"
	@docker-compose exec backend uvicorn src.main:app --reload

# データベース操作
db-migrate: ## データベースマイグレーションを実行
	@echo "$(CYAN)データベースマイグレーションを実行しています...$(RESET)"
	@docker-compose exec backend alembic upgrade head

db-rollback: ## データベースマイグレーションをロールバック
	@echo "$(CYAN)データベースマイグレーションをロールバックしています...$(RESET)"
	@docker-compose exec backend alembic downgrade -1

# テスト
test: test-backend test-frontend test-ml ## 全てのテストを実行

test-backend: ## バックエンドのテストを実行
	@echo "$(CYAN)バックエンドのテストを実行しています...$(RESET)"
	@docker-compose exec backend pytest

test-frontend: ## フロントエンドのテストを実行
	@echo "$(CYAN)フロントエンドのテストを実行しています...$(RESET)"
	@docker-compose exec frontend npm test

test-ml: ## MLサービスのテストを実行
	@echo "$(CYAN)MLサービスのテストを実行しています...$(RESET)"
	@docker-compose exec ml_service pytest

# リント
lint: lint-backend lint-frontend ## 全てのリントを実行

lint-backend: ## バックエンドのリントを実行
	@echo "$(CYAN)バックエンドのリントを実行しています...$(RESET)"
	@docker-compose exec backend flake8

lint-frontend: ## フロントエンドのリントを実行
	@echo "$(CYAN)フロントエンドのリントを実行しています...$(RESET)"
	@docker-compose exec frontend npm run lint

# ログ
logs: ## サービスのログを表示
	@docker-compose logs -f

# クリーンアップ
clean: ## 不要なファイルとコンテナを削除
	@echo "$(YELLOW)クリーンアップを実行しています...$(RESET)"
	@docker-compose down -v
	@find . -type d -name "__pycache__" -exec rm -r {} +
	@find . -type d -name ".pytest_cache" -exec rm -r {} +
	@find . -type d -name "node_modules" -exec rm -r {} +
	@find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)クリーンアップが完了しました$(RESET)"

# ヘルプ
help: ## このヘルプメッセージを表示
	@echo "使用可能なコマンド:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
