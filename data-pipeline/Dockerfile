# 開発環境
FROM bitnami/spark:3.5.0 AS development

WORKDIR /app

# システムの依存関係
USER root
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# ソースコードのコピー
COPY . .

# 開発用のエントリーポイント
CMD ["python3", "streaming/stream_processor.py"]

# 本番環境
FROM bitnami/spark:3.5.0 AS production

WORKDIR /app

# システムの依存関係
USER root
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# ソースコードのコピー
COPY . .

# 本番用のエントリーポイント
CMD ["python3", "streaming/stream_processor.py"]