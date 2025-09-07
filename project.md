# Non-Cog Learning Platform

## プロジェクト概要

Non-Cog は、生徒の非認知能力を育成するための革新的な学習プラットフォームです。AI を活用した適応型学習システムと、ゲーム形式の課題を通じて、生徒の「やり抜く力」や「協調性」などの非認知能力を効果的に育成します。

## 開発環境のセットアップ

### 前提条件

- Docker
- Docker Compose
- Git

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

3. Docker コンテナの起動

```bash
# 全サービスの起動
docker-compose up -d

# 特定のサービスのみ起動
docker-compose up -d frontend backend
```

4. 開発用コマンド

```bash
# フロントエンドの開発サーバー
docker-compose exec frontend npm run dev

# バックエンドの開発サーバー
docker-compose exec backend uvicorn src.main:app --reload

# MLサービスの開発
docker-compose exec ml_service python -m pytest

# データパイプラインの開発
docker-compose exec data_pipeline python stream_processor.py
```

### 各サービスのアクセス

- フロントエンド: http://localhost:3000
- バックエンド API: http://localhost:8000
- ML サービス: http://localhost:8001
- MLflow: http://localhost:5000
- Spark UI: http://localhost:8080
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

## プロジェクト構造

```
.
├── frontend/           # フロントエンドアプリケーション
│   └── src/
│       ├── components/ # Reactコンポーネント
│       ├── pages/      # ページコンポーネント
│       ├── styles/     # スタイルシート
│       └── utils/      # ユーティリティ関数
│
├── backend/            # バックエンドアプリケーション
│   └── src/
│       ├── api/       # APIエンドポイント
│       ├── core/      # ビジネスロジック
│       ├── domain/    # ドメインモデル
│       └── infrastructure/ # インフラストラクチャ層
│
├── ml/                # 機械学習モデル
│   ├── models/       # モデル定義
│   ├── training/     # 学習スクリプト
│   ├── evaluation/   # 評価スクリプト
│   └── federated/    # フェデレーテッドラーニング
│
└── data-pipeline/     # データ処理パイプライン
    ├── kafka/        # Kafkaプロデューサー/コンシューマー
    ├── spark/        # Sparkジョブ
    └── streaming/    # ストリーミング処理
```

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

### データパイプライン

- Apache Kafka
- Apache Spark
- Spark Streaming

### MLOps

- MLflow
- Kubeflow
- Docker
- Kubernetes

## 開発ガイドライン

### コーディング規約

- Python: PEP 8
- TypeScript: ESLint + Prettier
- コメントは日本語で記述

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

### テスト

```bash
# フロントエンドのテスト
docker-compose exec frontend npm test

# バックエンドのテスト
docker-compose exec backend pytest

# MLモデルのテスト
docker-compose exec ml_service pytest

# データパイプラインのテスト
docker-compose exec data_pipeline pytest
```

### デバッグ

- フロントエンド: Chrome DevTools
- バックエンド: デバッガー（VSCode）
- ML: TensorBoard
- データパイプライン: Spark UI

## トラブルシューティング

### よくある問題

1. コンテナが起動しない

```bash
# ログの確認
docker-compose logs [service_name]

# コンテナの再起動
docker-compose restart [service_name]
```

2. パッケージの依存関係エラー

```bash
# パッケージの更新
docker-compose exec [service_name] pip install -r requirements.txt
```

3. データベース接続エラー

```bash
# DBコンテナの状態確認
docker-compose ps db
docker-compose logs db
```

### デバッグツール

- Prometheus: メトリクス監視
- Grafana: 可視化
- MLflow: 実験管理

## 貢献ガイド

1. Issue の作成
2. ブランチの作成
3. 開発とテスト
4. PR の作成
5. レビュー
6. マージ

## ライセンス

MIT License
