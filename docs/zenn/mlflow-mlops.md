---
title: "MLflowで実験トラッキングしてみた"
emoji: "🧪"
type: "tech"
topics: ["mlflow", "mlops", "kubernetes", "gke", "ml"]
published: false
---

機械学習の実験、Excel やメモで管理してると地獄…ということで MLflow を“使ってみた”記録です。パラメータ・メトリクス・モデルをいい感じに残せます。

## セットアップ

- GKE 上に Tracking Server（`k8s/monitoring.yaml`）
- Ingress で `mlflow.<IP>.nip.io` を公開

> デモ画像差し込みポイント
>
> ```md
> ![MLflow UIのExperiment一覧](./images/mlflow_experiments.png)
> ```

## ロギング例

```python
import mlflow
mlflow.set_tracking_uri("https://mlflow.<IP>.nip.io")
mlflow.set_experiment("noncog-exp")
with mlflow.start_run(run_name="trial-001"):
    mlflow.log_param("model", "xgboost")
    mlflow.log_param("max_depth", 6)
    mlflow.log_metric("val_auc", 0.842)
    mlflow.log_artifact("./models/model.pkl")
```

> デモ画像差し込みポイント
>
> ```md
> ![Run詳細画面（メトリクス推移）](./images/mlflow_run.png)
> ```

## コツ

- URI/トークンは環境変数
- アーティファクトは GCS/S3
- 実験名とタグの命名をチームで統一

## まとめ

「ちゃんと再現できる」状態は強いです。UI のスクショを貼って、定量的に語れるようにしておくと就活で刺さります。
