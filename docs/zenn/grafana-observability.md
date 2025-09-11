---
title: "Grafanaでメトリクス可視化してみた（GKE/アプリ監視）"
emoji: "📊"
type: "tech"
topics: ["grafana", "prometheus", "kubernetes", "gke", "observability"]
published: false
---

GKE で動かしてるアプリの状態、ちゃんと見えないと怖い…ということで Grafana + Prometheus を“入れてみた”手順をまとめました。最小構成でダッシュボード表示まで。

## やること

- Prometheus でメトリクス収集
- Grafana で可視化（ダッシュボード）
- FastAPI のメトリクスも拾えるようにする

> デモ画像差し込みポイント
>
> ```md
> ![Grafanaのホーム画面](./images/grafana_home.png)
> ```

## デプロイ

`k8s/monitoring.yaml` に Grafana/Prometheus を定義して適用します。

```bash
kubectl apply -f k8s/monitoring.yaml -n noncog
kubectl rollout status deployment/grafana -n noncog
kubectl rollout status deployment/prometheus -n noncog
```

Ingress には `grafana.<IP>.nip.io` / `prometheus.<IP>.nip.io` を追加。

> デモ画像差し込みポイント
>
> ```md
> ![PrometheusのTargetsがUPになっている様子](./images/prometheus_targets.png)
> ```

## Grafana 設定のコツ

- 初期ユーザー `admin` / `admin` → すぐ変更
- Data Source に Prometheus を登録（Service の URL 指定）

## 使えるダッシュボード例

- Kubernetes / Compute Resources / Namespace（ID: 315）
- Kubernetes / API server（ID: 12006）
- FastAPI カスタムメトリクス

### FastAPI のメトリクス

```python
from prometheus_client import Counter, Histogram
REQUEST_COUNT = Counter('http_requests_total', 'HTTP requests', ['method', 'endpoint', 'http_status'])
LATENCY = Histogram('http_request_latency_seconds', 'Request latency', ['endpoint'])
```

Grafana クエリ例：

```
sum(rate(http_requests_total{endpoint="/api/login"}[5m])) by (http_status)
```

> デモ画像差し込みポイント
>
> ```md
> ![FastAPIメトリクスのダッシュボード例](./images/grafana_fastapi.png)
> ```

## ハマりどころ

- ダッシュボードが空 →Prometheus Targets をチェック
- 401/403→Ingress/認証周りの設定
- 重い →retention やサンプリング間隔を調整

## まとめ

“とりあえず見える化”ができると、障害対応もかなり楽になります。キャプチャをポートフォリオに貼って、運用力もアピールしましょう！
