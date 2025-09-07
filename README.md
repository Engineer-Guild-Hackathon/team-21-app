# ２１　チーム名：Error404 TeamName NotFound

このリポジトリはハッカソン提出用の雛形です。以下の項目をすべて埋めてください。

---

## チーム情報

- チーム番号: （21）
- チーム名: （Error404 TeamName NotFound）
- プロダクト名: （Non-Cog Learning Platform）
- メンバー: （GitHub アカウントまたは名前を列挙）

---

## デモ　/ プレゼン資料

- デモ URL:
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

- Docker
- Kubernetes
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

詳細な技術仕様とガイドラインについては、[project.md](./project.md)を参照してください。

## ライセンス

MIT License
