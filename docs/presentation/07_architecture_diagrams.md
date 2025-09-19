# 7. アーキテクチャ図（Mermaid）

## 7.1 システム全体アーキテクチャ

### 7.1s 簡易アーキテクチャ（発表用）

発表では下記の簡易図で主要要素のみを伝えます（ユーザー → フロント →Ingress→ バックエンド → データ層）。

```mermaid
graph TB
    U[ユーザー<br/>生徒/教師/保護者]
    FE[フロントエンド<br/>Next.js]
    IG[Ingress<br/>GKE LB]
    BE[バックエンド<br/>FastAPI]
    DB[(PostgreSQL)]
    RE[(Redis)]
    MON[監視<br/>Prometheus/Grafana]

    U --> FE --> IG --> BE
    BE --> DB
    BE --> RE
    BE --> MON
```

```mermaid
graph TB
    subgraph "ユーザー層"
        T[教師ポータル]
        S[生徒ポータル]
        P[保護者ポータル]
    end

    subgraph "フロントエンド層"
        FE[Next.js フロントエンド<br/>TypeScript + Tailwind]
    end

    subgraph "API ゲートウェイ"
        LB[ロードバランサー<br/>GKE Ingress]
    end

    subgraph "Kubernetes クラスター (GKE)"
        subgraph "バックエンドサービス"
            BE[FastAPI バックエンド<br/>Python 3.11]
            ML[ML サービス<br/>AI 分析]
        end

        subgraph "データ層"
            DB[(PostgreSQL<br/>Cloud SQL)]
            REDIS[(Redis<br/>キャッシュ・セッション)]
            KAFKA[Apache Kafka<br/>ストリーミングデータ]
        end
    end

    subgraph "外部サービス"
        GEMINI[Google Gemini API<br/>自然言語処理・AI]
        STORAGE[Google Cloud Storage<br/>静的アセット]
        BUILD[Google Cloud Build<br/>CI/CD]
    end

    subgraph "監視・運用"
        MON[Prometheus + Grafana<br/>メトリクス・ダッシュボード]
        LOG[構造化ログ<br/>集約ログ]
    end

    T --> FE
    S --> FE
    P --> FE

    FE --> LB
    LB --> BE
    LB --> ML

    BE --> DB
    BE --> REDIS
    BE --> KAFKA
    ML --> DB
    ML --> GEMINI

    KAFKA --> ML

    BUILD --> BE
    BUILD --> FE

    BE --> MON
    ML --> MON
    DB --> LOG
```

## 7.2 データフロー図

```mermaid
sequenceDiagram
    participant S as 生徒
    participant F as フロントエンド
    participant B as バックエンド
    participant M as MLサービス
    participant D as データベース
    participant K as Kafka
    participant G as Gemini API

    S->>F: クエスト開始
    F->>B: POST /api/quests/start
    B->>D: 進捗保存
    D-->>B: 進捗保存完了
    B-->>F: クエスト開始
    F-->>S: クエストUI更新

    S->>F: チャットメッセージ送信
    F->>B: POST /api/chat/messages
    B->>K: メッセージイベント発行
    K->>M: メッセージデータストリーミング
    M->>G: Geminiで分析
    G-->>M: 分析結果
    M->>D: 分析結果保存
    M-->>B: 洞察返却
    B-->>F: チャット応答
    F-->>S: 応答・洞察表示

    Note over M,D: リアルタイム分析
    Note over K,M: イベント駆動処理
```

## 7.3 マイクロサービス構成図

```mermaid
graph LR
    subgraph "フロントエンドサービス"
        AUTH_UI[認証UI]
        QUEST_UI[クエストUI]
        CHAT_UI[チャットUI]
        PROFILE_UI[プロフィールUI]
        TEACHER_UI[教師ダッシュボード]
    end

    subgraph "バックエンドサービス"
        AUTH_SVC[認証サービス<br/>JWT・ユーザー管理]
        QUEST_SVC[クエストサービス<br/>進捗・報酬]
        CHAT_SVC[チャットサービス<br/>セッション管理]
        USER_SVC[ユーザーサービス<br/>プロフィール・統計]
        AVATAR_SVC[アバターサービス<br/>カスタマイズ]
        CLASS_SVC[クラスサービス<br/>管理]
    end

    subgraph "MLサービス"
        NLP_SVC[自然言語処理サービス<br/>テキスト分析]
        SENTIMENT_SVC[感情分析サービス<br/>感情検出]
        PREDICT_SVC[予測サービス<br/>進捗予測]
    end

    subgraph "データサービス"
        DB_SVC[データベースサービス<br/>PostgreSQL]
        CACHE_SVC[キャッシュサービス<br/>Redis]
        STREAM_SVC[ストリームサービス<br/>Kafka]
    end

    AUTH_UI --> AUTH_SVC
    QUEST_UI --> QUEST_SVC
    CHAT_UI --> CHAT_SVC
    PROFILE_UI --> USER_SVC
    TEACHER_UI --> CLASS_SVC

    AUTH_SVC --> DB_SVC
    QUEST_SVC --> DB_SVC
    CHAT_SVC --> DB_SVC
    USER_SVC --> DB_SVC
    AVATAR_SVC --> DB_SVC
    CLASS_SVC --> DB_SVC

    CHAT_SVC --> STREAM_SVC
    STREAM_SVC --> NLP_SVC
    STREAM_SVC --> SENTIMENT_SVC
    STREAM_SVC --> PREDICT_SVC

    NLP_SVC --> CACHE_SVC
    SENTIMENT_SVC --> CACHE_SVC
    PREDICT_SVC --> CACHE_SVC
```

## 7.4 データベース設計図

```mermaid
erDiagram
    USERS {
        int id PK
        string email UK
        string hashed_password
        string full_name
        string role
        string class_id FK
        boolean is_active
        boolean is_verified
        boolean terms_accepted
        timestamp terms_accepted_at
        timestamp created_at
        timestamp updated_at
    }

    CLASSES {
        int id PK
        string name
        text description
        int teacher_id FK
        timestamp created_at
        timestamp updated_at
    }

    QUESTS {
        int id PK
        string title
        text description
        string quest_type
        string difficulty
        string target_skill
        int experience_points
        int coins
        int badge_id
        boolean is_active
        timestamp created_at
        timestamp updated_at
    }

    QUEST_PROGRESSES {
        int id PK
        int user_id FK
        int quest_id FK
        string status
        int current_step
        int total_steps
        decimal progress_percentage
        timestamp started_date
        timestamp completed_date
        timestamp created_at
        timestamp updated_at
    }

    AVATARS {
        int id PK
        string name
        text description
        string image_url
        string category
        string rarity
        boolean is_active
        int sort_order
        timestamp created_at
        timestamp updated_at
    }

    USER_AVATARS {
        int id PK
        int user_id FK
        int avatar_id FK
        boolean is_current
        timestamp acquired_at
    }

    USER_STATS {
        int id PK
        int user_id FK
        decimal grit_level
        decimal collaboration_level
        decimal self_regulation_level
        decimal emotional_intelligence_level
        timestamp created_at
        timestamp updated_at
    }

    CHAT_SESSIONS {
        int id PK
        int user_id FK
        string title
        string status
        timestamp created_at
        timestamp updated_at
    }

    CHAT_MESSAGES {
        int id PK
        int session_id FK
        string role
        text content
        jsonb analysis_result
        timestamp created_at
    }

    USERS ||--o{ CLASSES : "teaches"
    USERS ||--o{ QUEST_PROGRESSES : "has"
    QUESTS ||--o{ QUEST_PROGRESSES : "tracks"
    USERS ||--o{ USER_AVATARS : "owns"
    AVATARS ||--o{ USER_AVATARS : "owned_by"
    USERS ||--|| USER_STATS : "has"
    USERS ||--o{ CHAT_SESSIONS : "creates"
    CHAT_SESSIONS ||--o{ CHAT_MESSAGES : "contains"
```

## 7.5 セキュリティアーキテクチャ図

```mermaid
graph TB
    subgraph "外部アクセス"
        INTERNET[インターネット]
        CDN[Cloud CDN]
    end

    subgraph "セキュリティ層"
        WAF[Web Application Firewall]
        LB[SSL対応ロードバランサー]
        INGRESS[Kubernetes Ingress<br/>TLS終端]
    end

    subgraph "認証・認可"
        JWT[JWTトークン検証]
        RBAC[ロールベースアクセス制御]
        OAUTH[OAuth 2.0統合]
    end

    subgraph "アプリケーション層"
        FE[フロントエンド<br/>HTTPS専用]
        BE[バックエンドサービス<br/>内部通信]
        ML[MLサービス<br/>セキュアAPI]
    end

    subgraph "データ層セキュリティ"
        DB_ENCRYPT[(データベース<br/>保存時暗号化)]
        CACHE_SEC[(Redis<br/>認証+TLS)]
        KAFKA_SEC[Kafka<br/>SASL+SSL]
    end

    subgraph "監視・監査"
        AUDIT[監査ログ]
        MONITOR[セキュリティ監視]
        ALERTS[リアルタイムアラート]
    end

    INTERNET --> CDN
    CDN --> WAF
    WAF --> LB
    LB --> INGRESS

    INGRESS --> JWT
    JWT --> RBAC
    RBAC --> OAUTH

    OAUTH --> FE
    OAUTH --> BE
    BE --> ML

    BE --> DB_ENCRYPT
    BE --> CACHE_SEC
    ML --> KAFKA_SEC

    BE --> AUDIT
    ML --> AUDIT
    AUDIT --> MONITOR
    MONITOR --> ALERTS
```

## 7.6 CI/CD パイプライン図

```mermaid
graph LR
    subgraph "開発"
        DEV[開発者]
        GIT[Gitリポジトリ<br/>GitHub]
    end

    subgraph "CI/CDパイプライン"
        WEBHOOK[GitHub Webhook]
        ACTIONS[GitHub Actions]
        BUILD[Cloud Build]
        TEST[自動テスト]
        SECURITY[セキュリティスキャン]
    end

    subgraph "コンテナレジストリ"
        REGISTRY[Google Artifact Registry]
    end

    subgraph "デプロイメント"
        GKE[Google Kubernetes Engine]
        STAGING[ステージング環境]
        PROD[本番環境]
    end

    subgraph "監視"
        HEALTH[ヘルスチェック]
        METRICS[Prometheusメトリクス]
        LOGS[集約ログ]
        ALERTS[アラート管理]
    end

    DEV --> GIT
    GIT --> WEBHOOK
    WEBHOOK --> ACTIONS
    ACTIONS --> BUILD
    BUILD --> TEST
    TEST --> SECURITY
    SECURITY --> REGISTRY

    REGISTRY --> GKE
    GKE --> STAGING
    STAGING --> PROD

    PROD --> HEALTH
    PROD --> METRICS
    PROD --> LOGS
    METRICS --> ALERTS
    LOGS --> ALERTS
```

## 7.7 スケーラビリティ設計図

```mermaid
graph TB
    subgraph "負荷分散"
        CDN[Cloud CDN<br/>グローバル配信]
        LB[ロードバランサー<br/>トラフィック分散]
    end

    subgraph "オートスケーリング"
        HPA[水平Pod自動スケーラー<br/>CPU・メモリベース]
        VPA[垂直Pod自動スケーラー<br/>リソース最適化]
        CA[クラスター自動スケーラー<br/>ノード管理]
    end

    subgraph "アプリケーションスケーリング"
        FE_PODS[フロントエンドPod<br/>ステートレススケーリング]
        BE_PODS[バックエンドPod<br/>ステートレススケーリング]
        ML_PODS[MLサービスPod<br/>GPUスケーリング]
    end

    subgraph "データスケーリング"
        DB_READ[読み取りレプリカ<br/>クエリ分散]
        DB_SHARD[データベースシャーディング<br/>データ分割]
        CACHE_CLUSTER[Redisクラスター<br/>メモリスケーリング]
        KAFKA_PART[Kafkaパーティション<br/>メッセージスケーリング]
    end

    subgraph "監視・最適化"
        METRICS[メトリクス収集]
        OPTIMIZATION[自動最適化]
        PREDICTION[予測的スケーリング]
    end

    CDN --> LB
    LB --> HPA
    LB --> VPA
    LB --> CA

    HPA --> FE_PODS
    HPA --> BE_PODS
    VPA --> ML_PODS

    BE_PODS --> DB_READ
    BE_PODS --> DB_SHARD
    BE_PODS --> CACHE_CLUSTER
    BE_PODS --> KAFKA_PART

    FE_PODS --> METRICS
    BE_PODS --> METRICS
    ML_PODS --> METRICS

    METRICS --> OPTIMIZATION
    OPTIMIZATION --> PREDICTION
    PREDICTION --> HPA
```

## 7.8 非認知能力分析フロー図

```mermaid
graph TD
    subgraph "データ収集"
        QUEST[クエストインタラクション]
        CHAT[チャットメッセージ]
        BEHAVIOR[行動パターン]
        TIME[学習時間]
    end

    subgraph "データ処理"
        NLP[自然言語処理]
        SENTIMENT[感情分析]
        PATTERN[パターン認識]
        METRICS[行動メトリクス]
    end

    subgraph "AI分析"
        GEMINI[Google Gemini API]
        ML_MODELS[機械学習モデル]
        STATISTICS[統計分析]
    end

    subgraph "スキル評価"
        GRIT[やり抜く力レベル]
        COLLAB[協働性レベル]
        SELF_REG[自己制御レベル]
        EMOTIONAL[情動知能レベル]
    end

    subgraph "洞察・推奨"
        INSIGHTS[個別化洞察]
        RECOMMENDATIONS[学習推奨]
        PROGRESS[進捗追跡]
        PREDICTIONS[将来予測]
    end

    QUEST --> NLP
    CHAT --> NLP
    BEHAVIOR --> PATTERN
    TIME --> METRICS

    NLP --> GEMINI
    SENTIMENT --> GEMINI
    PATTERN --> ML_MODELS
    METRICS --> STATISTICS

    GEMINI --> GRIT
    GEMINI --> COLLAB
    ML_MODELS --> SELF_REG
    STATISTICS --> EMOTIONAL

    GRIT --> INSIGHTS
    COLLAB --> RECOMMENDATIONS
    SELF_REG --> PROGRESS
    EMOTIONAL --> PREDICTIONS

    INSIGHTS --> RECOMMENDATIONS
    RECOMMENDATIONS --> PROGRESS
    PROGRESS --> PREDICTIONS
```

## 7.9 ゲーミフィケーション要素図

```mermaid
graph LR
    subgraph "ユーザーエンゲージメント"
        AVATAR[アバターシステム]
        LEVELS[レベルシステム]
        BADGES[バッジシステム]
        REWARDS[報酬システム]
    end

    subgraph "クエストメカニクス"
        QUESTS[クエストタイプ]
        DIFFICULTY[難易度レベル]
        PROGRESS[進捗追跡]
        COMPLETION[完了報酬]
    end

    subgraph "ソーシャル機能"
        FRIENDS[フレンドシステム]
        LEADERBOARD[リーダーボード]
        SHARING[進捗共有]
        COLLAB[協力クエスト]
    end

    subgraph "モチベーションシステム"
        ACHIEVEMENTS[実績システム]
        STREAKS[学習ストリーク]
        CHALLENGES[デイリーチャレンジ]
        EVENTS[特別イベント]
    end

    AVATAR --> LEVELS
    LEVELS --> BADGES
    BADGES --> REWARDS

    QUESTS --> DIFFICULTY
    DIFFICULTY --> PROGRESS
    PROGRESS --> COMPLETION

    FRIENDS --> LEADERBOARD
    LEADERBOARD --> SHARING
    SHARING --> COLLAB

    ACHIEVEMENTS --> STREAKS
    STREAKS --> CHALLENGES
    CHALLENGES --> EVENTS

    REWARDS --> ACHIEVEMENTS
    COMPLETION --> STREAKS
    COLLAB --> CHALLENGES
    EVENTS --> AVATAR
```

これらの Mermaid 図により、Non-Cog アプリケーションの複雑なアーキテクチャとデータフローを視覚的に理解しやすく表現できます。プレゼンテーションやドキュメントで活用してください。
