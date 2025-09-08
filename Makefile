.PHONY: setup start stop restart clean test lint logs help install-dev-deps format

# デフォルトのターゲット
.DEFAULT_GOAL := help

# 色の定義
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Python仮想環境の設定
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# 環境セットアップ
setup: create-venv install-dev-deps ## 開発環境のセットアップ
	@echo "$(CYAN)環境をセットアップしています...$(RESET)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN).envファイルを作成しました$(RESET)"; \
	fi
	@docker-compose build
	@echo "$(GREEN)セットアップが完了しました$(RESET)"

# 仮想環境の作成
create-venv: ## Python仮想環境を作成
	@echo "$(CYAN)Python仮想環境を作成しています...$(RESET)"
	@python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@echo "$(GREEN)Python仮想環境を作成しました$(RESET)"

# 開発用依存関係のインストール
install-dev-deps: install-backend-deps install-frontend-deps install-ml-deps ## 全ての開発用依存関係をインストール

install-backend-deps: ## バックエンドの開発用依存関係をインストール
	@echo "$(CYAN)バックエンドの開発用依存関係をインストールしています...$(RESET)"
	@cd backend && $(PIP) install -r requirements.txt
	@$(PIP) install black isort mypy flake8 pytest pytest-cov
	@echo "$(GREEN)バックエンドの開発用依存関係をインストールしました$(RESET)"

install-frontend-deps: ## フロントエンドの開発用依存関係をインストール
	@echo "$(CYAN)フロントエンドの開発用依存関係をインストールしています...$(RESET)"
	@cd frontend && npm install
	@echo "$(GREEN)フロントエンドの開発用依存関係をインストールしました$(RESET)"

install-ml-deps: ## MLサービスの開発用依存関係をインストール
	@echo "$(CYAN)MLサービスの開発用依存関係をインストールしています...$(RESET)"
	@cd ml && $(PIP) install -r requirements.txt
	@$(PIP) install black isort mypy flake8 pytest pytest-cov
	@echo "$(GREEN)MLサービスの開発用依存関係をインストールしました$(RESET)"

# フォーマット
format: format-backend format-frontend format-ml ## 全てのコードを自動フォーマット

format-backend: ## バックエンドのコードを自動フォーマット
	@echo "$(CYAN)バックエンドのコードをフォーマットしています...$(RESET)"
	@cd backend && $(VENV)/bin/black .
	@cd backend && $(VENV)/bin/isort .
	@echo "$(GREEN)バックエンドのコードをフォーマットしました$(RESET)"

format-frontend: ## フロントエンドのコードを自動フォーマット
	@echo "$(CYAN)フロントエンドのコードをフォーマットしています...$(RESET)"
	@cd frontend && npm run format
	@cd frontend && npm run lint:fix
	@echo "$(GREEN)フロントエンドのコードをフォーマットしました$(RESET)"

format-ml: ## MLサービスのコードを自動フォーマット
	@echo "$(CYAN)MLサービスのコードをフォーマットしています...$(RESET)"
	@cd ml && $(VENV)/bin/black .
	@cd ml && $(VENV)/bin/isort .
	@echo "$(GREEN)MLサービスのコードをフォーマットしました$(RESET)"

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
	@cd frontend && npm run dev

dev-backend: ## バックエンド開発サーバーを起動
	@echo "$(CYAN)バックエンド開発サーバーを起動しています...$(RESET)"
	@cd backend && $(PYTHON) -m uvicorn src.main:app --reload

# データベース操作
db-migrate: ## データベースマイグレーションを実行
	@echo "$(CYAN)データベースマイグレーションを実行しています...$(RESET)"
	@cd backend && alembic upgrade head

db-rollback: ## データベースマイグレーションをロールバック
	@echo "$(CYAN)データベースマイグレーションをロールバックしています...$(RESET)"
	@cd backend && alembic downgrade -1

# テスト
test: test-backend test-frontend test-ml ## 全てのテストを実行

test-backend: ## バックエンドのテストを実行
	@echo "$(CYAN)バックエンドのテストを実行しています...$(RESET)"
	@cd backend && $(PYTHON) -m pytest

test-frontend: ## フロントエンドのテストを実行
	@echo "$(CYAN)フロントエンドのテストを実行しています...$(RESET)"
	@cd frontend && npm test

test-ml: ## MLサービスのテストを実行
	@echo "$(CYAN)MLサービスのテストを実行しています...$(RESET)"
	@cd ml && $(PYTHON) -m pytest

# リント
lint: lint-check lint-fix ## 全てのリントチェックと自動修正を実行

lint-check: lint-check-backend lint-check-frontend lint-check-ml ## 全てのリントチェックを実行

lint-fix: lint-fix-backend lint-fix-frontend lint-fix-ml ## 全ての自動修正可能なリント問題を修正

# バックエンドのリント
lint-check-backend: ## バックエンドのリントチェックを実行
	@echo "$(CYAN)バックエンドのリントチェックを実行しています...$(RESET)"
	@cd backend && $(VENV)/bin/black . --check
	@cd backend && $(VENV)/bin/isort . --check-only
	@cd backend && $(VENV)/bin/mypy .
	@cd backend && $(VENV)/bin/flake8

lint-fix-backend: ## バックエンドのリント自動修正を実行
	@echo "$(CYAN)バックエンドのリント自動修正を実行しています...$(RESET)"
	@cd backend && $(VENV)/bin/black .
	@cd backend && $(VENV)/bin/isort .

# フロントエンドのリント
lint-check-frontend: ## フロントエンドのリントチェックを実行
	@echo "$(CYAN)フロントエンドのリントチェックを実行しています...$(RESET)"
	@cd frontend && npm run lint
	@cd frontend && npm run type-check

lint-fix-frontend: ## フロントエンドのリント自動修正を実行
	@echo "$(CYAN)フロントエンドのリント自動修正を実行しています...$(RESET)"
	@cd frontend && npm run lint:fix
	@cd frontend && npm run format

# MLサービスのリント
lint-check-ml: ## MLサービスのリントチェックを実行
	@echo "$(CYAN)MLサービスのリントチェックを実行しています...$(RESET)"
	@cd ml && $(VENV)/bin/black . --check
	@cd ml && $(VENV)/bin/isort . --check-only
	@cd ml && $(VENV)/bin/mypy .
	@cd ml && $(VENV)/bin/flake8

lint-fix-ml: ## MLサービスのリント自動修正を実行
	@echo "$(CYAN)MLサービスのリント自動修正を実行しています...$(RESET)"
	@cd ml && $(VENV)/bin/black .
	@cd ml && $(VENV)/bin/isort .

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
	@rm -rf $(VENV)
	@echo "$(GREEN)クリーンアップが完了しました$(RESET)"

# ヘルプ
help: ## このヘルプメッセージを表示
	@echo "使用可能なコマンド:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'