.PHONY: setup start stop restart clean test lint logs help format

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
	@docker-compose build --no-cache
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
	@docker-compose exec backend uvicorn src.main:app --reload --host 0.0.0.0

# データベース操作
db-migrate: ## データベースマイグレーションを実行
	@echo "$(CYAN)データベースマイグレーションを実行しています...$(RESET)"
	@docker-compose run --rm backend /bin/sh -c "\
		until pg_isready -h db -U postgres; do \
			echo 'Waiting for database to be ready...' && sleep 2; \
		done && \
		until curl --output /dev/null --silent --fail http://backend:8000/docs; do \
			echo 'Waiting for backend to be ready...' && sleep 2; \
		done && \
		alembic upgrade head"
	@echo "$(GREEN)マイグレーションが完了しました$(RESET)"

db-rollback: ## データベースマイグレーションをロールバック
	@echo "$(CYAN)データベースマイグレーションをロールバックしています...$(RESET)"
	@docker-compose exec backend alembic downgrade -1

db-seed: ## データベースに初期データを投入
	@echo "$(CYAN)初期データを投入しています...$(RESET)"
	@docker-compose exec backend python scripts/seed_data.py
	@echo "$(GREEN)初期データの投入が完了しました$(RESET)"

db-setup: db-migrate db-seed ## データベースのセットアップ（マイグレーション + シードデータ）
	@echo "$(GREEN)データベースのセットアップが完了しました$(RESET)"

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

# フォーマット
format: ## 全てのコードを自動フォーマット
	@echo "$(CYAN)フロントエンドのコードをフォーマットしています...$(RESET)"
	@docker-compose exec frontend sh -c "npm run format && npm run lint:fix"
	@echo "$(GREEN)フロントエンドのコードをフォーマットしました$(RESET)"
	
	@echo "$(CYAN)MLサービスのコードをフォーマットしています...$(RESET)"
	@docker-compose exec ml_service sh -c "black . && isort ."
	@echo "$(GREEN)MLサービスのコードをフォーマットしました$(RESET)"
	
	@echo "$(CYAN)バックエンドのコードをフォーマットしています...$(RESET)"
	@docker-compose exec backend sh -c "black . && isort ."
	@echo "$(GREEN)バックエンドのコードをフォーマットしました$(RESET)"

format-frontend: ## フロントエンドのコードを自動フォーマット
	@echo "$(CYAN)フロントエンドのコードをフォーマットしています...$(RESET)"
	@docker-compose exec frontend sh -c "npm run format && npm run lint:fix"
	@echo "$(GREEN)フロントエンドのコードをフォーマットしました$(RESET)"

format-backend: ## バックエンドのコードを自動フォーマット
	@echo "$(CYAN)バックエンドのコードをフォーマットしています...$(RESET)"
	@docker-compose exec backend sh -c "black . && isort ."
	@echo "$(GREEN)バックエンドのコードをフォーマットしました$(RESET)"

format-ml: ## MLサービスのコードを自動フォーマット
	@echo "$(CYAN)MLサービスのコードをフォーマットしています...$(RESET)"
	@docker-compose exec ml_service sh -c "black . && isort ."
	@echo "$(GREEN)MLサービスのコードをフォーマットしました$(RESET)"

# リント
lint: lint-check lint-fix ## 全てのリントチェックと自動修正を実行

lint-check: lint-check-backend lint-check-frontend lint-check-ml ## 全てのリントチェックを実行

lint-fix: lint-fix-backend lint-fix-frontend lint-fix-ml ## 全ての自動修正可能なリント問題を修正

# バックエンドのリント
lint-check-backend: ## バックエンドのリントチェックを実行
	@echo "$(CYAN)バックエンドのリントチェックを実行しています...$(RESET)"
	@docker-compose exec backend sh -c "black . --check && isort . --check-only && mypy . && flake8"

lint-fix-backend: ## バックエンドのリント自動修正を実行
	@echo "$(CYAN)バックエンドのリント自動修正を実行しています...$(RESET)"
	@docker-compose exec backend sh -c "black . && isort ."

# フロントエンドのリント
lint-check-frontend: ## フロントエンドのリントチェックを実行
	@echo "$(CYAN)フロントエンドのリントチェックを実行しています...$(RESET)"
	@docker-compose exec frontend sh -c "npm run lint && npm run type-check"

lint-fix-frontend: ## フロントエンドのリント自動修正を実行
	@echo "$(CYAN)フロントエンドのリント自動修正を実行しています...$(RESET)"
	@docker-compose exec frontend sh -c "npm run lint:fix && npm run format"

# MLサービスのリント
lint-check-ml: ## MLサービスのリントチェックを実行
	@echo "$(CYAN)MLサービスのリントチェックを実行しています...$(RESET)"
	@docker-compose exec ml_service sh -c "black . --check && isort . --check-only && mypy . && flake8"

lint-fix-ml: ## MLサービスのリント自動修正を実行
	@echo "$(CYAN)MLサービスのリント自動修正を実行しています...$(RESET)"
	@docker-compose exec ml_service sh -c "black . && isort ."

# ログ
logs: ## サービスのログを表示
	@docker-compose logs -f

# クリーンアップ
clean: ## 不要なファイルとコンテナを削除
	@echo "$(YELLOW)クリーンアップを実行しています...$(RESET)"
	@docker-compose down -v --remove-orphans
	@docker system prune -f
	@echo "$(GREEN)クリーンアップが完了しました$(RESET)"

# ヘルプ
help: ## このヘルプメッセージを表示
	@echo "使用可能なコマンド:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'