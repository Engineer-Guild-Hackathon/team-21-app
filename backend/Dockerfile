# 開発環境
FROM python:3.11-slim AS development

WORKDIR /app

# システムの依存関係
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# ソースコードのコピー
COPY . .

# PYTHONPATHの設定
ENV PYTHONPATH=/app

# 開発サーバーの起動
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# 本番環境
FROM python:3.11-slim AS production

WORKDIR /app

# システムの依存関係
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# ソースコードのコピー
COPY . .

# PYTHONPATHの設定
ENV PYTHONPATH=/app

# 本番サーバーの起動
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]