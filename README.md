# ２１　チーム名：Error404 TeamName NotFound

---

## チーム情報

- チーム番号: （21）
- チーム名: （Error404 TeamName NotFound）
- プロダクト名: （Non-Cog Learning Platform）
- メンバー: （・ロホマン シャヒン ・秦拓夢）

---

## デモ　/ プレゼン資料

- デモ URL: https://app.34.107.156.246.nip.io
- プレゼン URL：

---

# Non-Cog Learning Platform

## プロジェクト概要

Non-Cog は、生徒の非認知能力を育成するための革新的な学習プラットフォームです。AI を活用した適応型学習システムと、ゲーム形式の課題を通じて、生徒の「やり抜く力」や「協調性」などの非認知能力を効果的に育成します。

## 主な機能

- 個別最適化学習

  - AI が一人ひとりの学習進度や特性に合わせて、最適な学習コンテンツを提供
  - リアルタイムフィードバックによる学習支援
  - 進捗管理と可視化

- 感情分析による学習支援

  - 学習中の感情状態をリアルタイムで分析
  - モチベーション維持のためのアダプティブサポート
  - パーソナライズされたフィードバック

- 強化学習による最適化
  - 学習者の行動パターンの分析
  - 最適な学習パスの自動生成
  - 効果的な学習戦略の提案

## 本番環境

### デモアクセス

- **アプリケーション**: https://app.34.107.156.246.nip.io
- **API ドキュメント**: https://api.34.107.156.246.nip.io/docs

### デモアカウント

- **生徒**: 山田太郎 (student@example.com / password123)
- **保護者**: 山田花子 (parent@example.com / password123)
- **教師**: 佐藤先生 (teacher@example.com / password123)

## 開発環境のセットアップ

### 前提条件

- Docker
- Docker Compose
- Git
- Make

### セットアップ手順

1. リポジトリのクローン

```bash
git clone https://github.com/your-org/team-21-app.git
cd team-21-app
```

2. 環境変数の設定

```bash
cp .env.example .env
# .envファイルを編集して必要な環境変数を設定
```

3. 開発環境のセットアップと起動

```bash
# 環境のセットアップ（初回のみ）
make setup

# 全サービスの起動
make start
```

4. データベースのセットアップ

```bash
# マイグレーションの実行
make db-migrate
```

### 開発用コマンド

```bash
# フロントエンド開発サーバー
make dev-frontend

# バックエンド開発サーバー
make dev-backend

# テストの実行
make test

# リントの実行
make lint
```

### 各サービスのアクセス

#### 本番環境（GKE）

- フロントエンド: https://app.34.107.156.246.nip.io
- バックエンド API: https://api.34.107.156.246.nip.io
- API ドキュメント: https://api.34.107.156.246.nip.io/docs

#### 開発環境（ローカル）

- フロントエンド: http://localhost:3000
- バックエンド API: http://localhost:8000
- ML サービス: http://localhost:8001
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

## 技術スタック

### フロントエンド

- Next.js
- React
- TypeScript
- TailwindCSS

### バックエンド

- FastAPI
- Python
- PostgreSQL
- Redis

### 機械学習

- PyTorch
- TensorFlow
- DQN/A2C（強化学習）
- BERT（自然言語処理）

### インフラストラクチャ

#### 本番環境

- Google Kubernetes Engine (GKE)
- Google Cloud SQL (PostgreSQL)
- Google Artifact Registry
- Google Cloud Load Balancer
- SSL/TLS 証明書 (Managed Certificate)

#### 開発環境

- Docker
- Docker Compose
- Apache Kafka
- Apache Spark

## 開発ガイドライン

### Git ブランチ戦略

- main: 本番環境
- develop: 開発環境
- feature/\*: 機能開発
- bugfix/\*: バグ修正

### コミットメッセージ規約

- feat: 新機能
- fix: バグ修正
- docs: ドキュメント
- style: コードスタイル
- refactor: リファクタリング
- test: テストコード
- chore: ビルド・補助ツール

---

## アーキテクチャ概要

### 開発環境アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose                          │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Next.js) - Port 3000                           │
│  Backend (FastAPI) - Port 8000                            │
│  ML Service - Port 8001                                   │
│  PostgreSQL - Port 5432                                   │
│  Redis - Port 6379                                        │
│  Kafka - Port 9092                                        │
│  Spark Master - Port 8080                                 │
│  MLflow - Port 5000                                       │
│  Prometheus - Port 9090                                   │
│  Grafana - Port 3001                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## クリーンアーキテクチャ（開発環境/構成）

このプロジェクトはモノレポ構成です。レイヤ責務は概ね以下の通りです。

- frontend: UI/UX（Next.js App Router、`middleware.ts` で認可）
- backend: API（FastAPI）
  - `src/domain`: ドメインモデル/スキーマ/型
  - `src/services`: ユースケース（ドメインロジック）
  - `src/infrastructure`: DB 接続/リポジトリ/設定
  - `src/api/v1`: ルータ（入出力境界）
  - 認証は `src/core/security.py` と `api/v1/auth.py` に集約（JWT）
- ml: 補助的な ML サービス（将来の拡張を想定）

参考ディレクトリ

```
backend/
  src/
    api/v1/            # ルータ
    core/              # セキュリティ/共通
    domain/
      models/          # SQLAlchemy モデル
      schemas/         # Pydantic スキーマ
      types/           # 型・値オブジェクト
    infrastructure/
      database.py      # AsyncSession / Engine
      repositories/    # DB アクセス
    services/          # アプリケーションサービス
  alembic/
    versions/          # マイグレーション
```

### 非同期実装（バックエンド）

- DB は `AsyncSession` を使用。
- 認証フローは `/api/auth/token` → JWT 発行 → `/api/users/me` でユーザー取得。

### マイグレーション運用

- 生成: `alembic revision -m "message"`（コンテナ内）
- 適用: `make db-migrate` / `docker-compose exec backend alembic upgrade head`
- 破綻時は `upgrade heads` で分岐解消、最終手段として `make clean` で初期化。

---

## コード品質管理

### Lint / Format / Type Check / Test

- フロント（Next.js）
  - Format: Prettier（`make format-frontend`）
  - Lint: ESLint（`make lint-check-frontend`）
- バックエンド（FastAPI）
  - Format: Black / isort（`make format-backend`）
  - Lint: Flake8 / mypy（`make lint-check-backend`）
- ML サービス
  - Black / isort を `ml/requirements.txt` に記載。`make format-ml`
- テスト
  - まとめて: `make test`
  - 個別: `docker-compose exec backend pytest -q`

### 推奨: pre-commit フック

`pre-commit` を導入するとコミット前に自動で品質チェック/整形が走ります。

```
pip install pre-commit
pre-commit install
```

`.pre-commit-config.yaml` 例（抜粋）

```
- repo: https://github.com/psf/black
  rev: 24.4.2
  hooks:
    - id: black
- repo: https://github.com/pycqa/isort
  rev: 5.13.2
  hooks:
    - id: isort
```

### CI（任意）

- PR 時に Lint/TypeCheck/Test を実行するワークフローを `.github/workflows/ci.yml` へ配置推奨。

---
