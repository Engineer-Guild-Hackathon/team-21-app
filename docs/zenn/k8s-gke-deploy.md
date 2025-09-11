---
title: "GKEにデプロイしてみた話（FastAPI + Next.js）"
emoji: "🚀"
type: "tech"
topics: ["kubernetes", "gke", "gcp", "devops", "ingress", "fastapi", "nextjs"]
published: false
---

就活用に、実際にチーム開発でやったことを“やってみた”ベースでまとめました。FastAPI(バックエンド)と Next.js(フロント)を Docker 化して、Artifact Registry に push→GKE にデプロイ →Ingress で公開、まで一気通貫でやってます。

## この記事でやること

- コンテナをビルドしてレジストリへ push
- GKE へデプロイ（Deployment/Service/Ingress）
- ConfigMap/Secret で環境変数をいい感じに注入
- 動かないときの詰まったポイントも共有

## 全体図（イメージ）

- GKE（クラスタ）
- Artifact Registry（Docker イメージ保管）
- Cloud SQL（PostgreSQL） + Cloud SQL Proxy
- Ingress + ManagedCertificate（nip.io で HTTPS）

> デモ画像差し込みポイント
>
> ```md
> ![アーキテクチャ図](./images/k8s_arch.png)
> ```

## リポジトリ構成（抜粋）

- `k8s/backend.yaml`: Backend Deployment/Service
- `k8s/ingress.yaml`: Ingress + ManagedCertificate
- `k8s/config.yaml`: ConfigMap（API BASE や CORS）
- `k8s/monitoring.yaml`: Grafana/Prometheus/MLflow（おまけ）
- `backend/src/main.py`: CORS
- `backend/src/infrastructure/database.py`: DB 接続

## 事前準備

- `gcloud`ログイン＆プロジェクト設定
- Artifact Registry と GKE クラスタ作成
- Cloud SQL(PostgreSQL)作成＆DB 初期化

## ビルド & Push

```bash
# Backend
docker build -t REGION-docker.pkg.dev/PROJECT_ID/REPO/backend:TAG backend
docker push REGION-docker.pkg.dev/PROJECT_ID/REPO/backend:TAG

# Frontend
docker build -t REGION-docker.pkg.dev/PROJECT_ID/REPO/frontend:TAG frontend
docker push REGION-docker.pkg.dev/PROJECT_ID/REPO/frontend:TAG
```

> デモ画像差し込みポイント
>
> ```md
> ![Artifact Registryにpushできた画面](./images/artifact_registry.png)
> ```

## Kubernetes 反映

```bash
kubectl create ns noncog || true
kubectl apply -f k8s/config.yaml -n noncog
kubectl apply -f k8s/backend.yaml -n noncog
kubectl rollout status deployment/backend -n noncog
kubectl apply -f k8s/frontend.yaml -n noncog
kubectl rollout status deployment/frontend -n noncog
kubectl apply -f k8s/ingress.yaml -n noncog
```

- `backend.yaml`のポイント
  - `ALLOWED_ORIGINS` で CORS 制御
  - `DB_PASS`を`DATABASE_URL`より前に置く（展開順の落とし穴）

> デモ画像差し込みポイント
>
> ```md
> ![kubectl rolloutの成功ログ](./images/kubectl_rollout.png)
> ```

## 動作確認

- フロント: `https://app.<IP>.nip.io`
- API: `https://api.<IP>.nip.io`

> デモ画像差し込みポイント
>
> ```md
> ![IngressでHTTPSアクセスOK](./images/ingress_ok.png)
> ```

## 詰まったポイント（実録）

- DB 接続 500: `DATABASE_URL`の展開順でパスワード入ってなかった → 順序修正
- Cloud SQL Admin API が無効 → コンソールで有効化
- Cloud SQL が`PENDING_CREATE`→`RUNNABLE`まで待つ
- DB`noncog`未作成 →`gcloud sql databases create noncog`
- スキーマ不一致 →`users.password_hash`→`hashed_password`へ、欠損カラム追加
- CORS→`ALLOWED_ORIGINS`と`ENVIRONMENT`連動の`get_allowed_origins()`を確認

## まとめ

GKE に“デプロイしてみた”のリアルな手順でした。詰まりどころも含めて、面接で「実装のどこが難しかったか」を語れる材料になります！

> 画像について
>
> - ローカルでは `docs/zenn/images/*.png` に置いてプレビュー
> - Zenn に投稿するときは Zenn の画像アップロード機能で置き換えが無難です。
