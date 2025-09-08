# Non-Cog Learning Platform システムアーキテクチャ

## システム概要図

```mermaid
graph TB
    subgraph "フロントエンド"
        UI[Webインターフェース]
        Auth[認証コンポーネント]
        Chat[チャットUI]
        Progress[進捗管理UI]
        Feedback[フィードバックUI]
    end

    subgraph "バックエンド"
        API[FastAPI]
        Auth_Service[認証サービス]
        Chat_Service[チャットサービス]
        Progress_Service[進捗管理サービス]
        Feedback_Service[フィードバックサービス]
    end

    subgraph "データストア"
        DB[(PostgreSQL)]
        Cache[(Redis)]
        Queue[Apache Kafka]
    end

    subgraph "機械学習サービス"
        ML_Service[MLサービス]
        Emotion_Analysis[感情分析]
        Learning_Optimization[学習最適化]
        subgraph "モデル"
            BERT[BERT NLP]
            DQN[強化学習 DQN/A2C]
        end
    end

    subgraph "データパイプライン"
        Spark[Apache Spark]
        MLflow[MLflow]
    end

    subgraph "モニタリング"
        Prometheus[Prometheus]
        Grafana[Grafana]
    end

    %% フロントエンドの接続
    UI --> Auth
    UI --> Chat
    UI --> Progress
    UI --> Feedback

    %% バックエンドとの接続
    Auth --> API
    Chat --> API
    Progress --> API
    Feedback --> API

    %% バックエンドサービスの接続
    API --> Auth_Service
    API --> Chat_Service
    API --> Progress_Service
    API --> Feedback_Service

    %% データストアとの接続
    Auth_Service --> DB
    Chat_Service --> DB
    Progress_Service --> DB
    Feedback_Service --> DB

    Auth_Service --> Cache
    Chat_Service --> Cache

    Chat_Service --> Queue
    Progress_Service --> Queue

    %% 機械学習サービスとの接続
    Queue --> ML_Service
    ML_Service --> Emotion_Analysis
    ML_Service --> Learning_Optimization

    Emotion_Analysis --> BERT
    Learning_Optimization --> DQN

    %% データパイプラインの接続
    ML_Service --> Spark
    ML_Service --> MLflow

    %% モニタリングの接続
    API --> Prometheus
    ML_Service --> Prometheus
    Prometheus --> Grafana
```

## データフロー図

```mermaid
sequenceDiagram
    actor User
    participant Frontend
    participant Backend
    participant Cache
    participant DB
    participant ML
    participant Queue

    User->>Frontend: 学習開始
    Frontend->>Backend: セッション開始リクエスト
    Backend->>DB: ユーザー情報取得
    Backend->>Cache: セッション情報保存
    Backend->>Frontend: セッション開始応答

    loop 学習プロセス
        User->>Frontend: 入力/アクション
        Frontend->>Backend: イベント送信
        Backend->>Queue: イベントパブリッシュ
        Queue->>ML: イベント処理
        ML->>DB: 分析結果保存
        ML->>Queue: レスポンス生成
        Queue->>Backend: レスポンス取得
        Backend->>Frontend: UI更新
        Frontend->>User: フィードバック表示
    end

    User->>Frontend: 学習終了
    Frontend->>Backend: セッション終了リクエスト
    Backend->>DB: 学習結果保存
    Backend->>Cache: セッション情報クリア
    Backend->>Frontend: 終了確認
```

## コンポーネント構成

### フロントエンド（Next.js）

- **認証（Auth）**: ユーザー認証・認可管理
- **チャット**: AI キャラクターとのインタラクション
- **進捗管理**: 学習状況の可視化
- **フィードバック**: 学習者へのフィードバック表示

### バックエンド（FastAPI）

- **認証サービス**: JWT 認証、セッション管理
- **チャットサービス**: メッセージ処理、感情分析連携
- **進捗管理サービス**: 学習データ管理、最適化
- **フィードバックサービス**: フィードバック生成

### 機械学習サービス

- **感情分析**: BERT モデルによるテキスト感情分析
- **学習最適化**: DQN/A2C による学習パス最適化
- **MLflow**: モデルのバージョン管理、実験管理

### データストア

- **PostgreSQL**: ユーザーデータ、学習履歴
- **Redis**: セッション管理、キャッシュ
- **Kafka**: イベントストリーム、非同期処理

### データパイプライン

- **Apache Spark**: 大規模データ処理
- **MLflow**: モデル管理、デプロイメント

### モニタリング

- **Prometheus**: メトリクス収集
- **Grafana**: 可視化、アラート

## 主要な処理フロー

1. **ユーザー認証フロー**

   - JWT ベースの認証
   - Redis でのセッション管理

2. **学習セッションフロー**

   - ユーザー入力の受付
   - リアルタイム感情分析
   - 学習パスの動的最適化

3. **フィードバックフロー**

   - 感情分析結果の活用
   - 学習進捗の分析
   - パーソナライズされたフィードバック生成

4. **データ分析フロー**
   - 学習データの収集
   - Spark による分析処理
   - モデルの継続的改善

## スケーラビリティと可用性

- Kubernetes 上での展開
- マイクロサービスアーキテクチャ
- 非同期処理によるスケーラビリティ確保
- キャッシュ層による高速化
