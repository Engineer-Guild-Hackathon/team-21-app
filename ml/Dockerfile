# 開発環境
FROM python:3.11-slim AS development

WORKDIR /app

# システム依存関係（必要に応じて拡張）
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係（存在しない場合は空のrequirementsでスキップされます）
COPY requirements.txt requirements.txt
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# ソースコード
COPY . .

# デフォルトのエントリポイント（開発時はdocker-compose側で上書き可能）
CMD ["python", "-m", "http.server", "8001"]
