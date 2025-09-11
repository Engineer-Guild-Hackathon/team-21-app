---
title: "Redisで爆速キャッシュやってみた"
emoji: "⚡"
type: "tech"
topics: ["redis", "caching", "fastapi", "nextjs", "kubernetes"]
published: false
---

APIのレスポンス、もっと速くならないかな？ということで Redis を“入れてみた”メモです。ホットデータをキャッシュしてDB負荷を減らす、定番だけど効きます。

## ざっくり構成
- Backend(FastAPI) → Redis → PostgreSQL
- フロントはISR/SSG/CSRと合わせてキャッシュ活用

> デモ画像差し込みポイント
> ```md
> ![キャッシュヒット率の可視化（Grafana例）](./images/redis_hits.png)
> ```

## FastAPIサンプル
```python
import aioredis, json
CACHE_TTL_SECONDS = 300

async def get_redis_pool():
    return await aioredis.from_url("redis://redis:6379", encoding="utf-8", decode_responses=True)

async def get_user_profile(user_id: str):
    r = await get_redis_pool()
    key = f"user:profile:{user_id}"
    cached = await r.get(key)
    if cached:
        return json.loads(cached)
    data = await fetch_from_db(user_id)
    await r.set(key, json.dumps(data), ex=CACHE_TTL_SECONDS)
    return data
```

## 設計のコツ
- TTLで自然に古くする＋書き込み時に該当キーをDEL
- 名前空間にバージョンを入れて一括無効化
- 再接続/タイムアウトは明示設定

## 監視
- Hit/Miss、レイテンシ、接続数をGrafanaで

## セキュリティ
- AUTHはSecretで管理、外部公開しない

## まとめ
まずはシンプルに始めて、効く箇所にだけキャッシュを当てるのがコスパ良いです。ベンチ結果のキャプチャを貼ると説得力アップ！
