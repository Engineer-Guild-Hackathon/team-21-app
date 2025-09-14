# ML Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY src/ ./src/

# ポートの公開
EXPOSE 8001

# アプリケーションの起動
CMD ["python", "-m", "uvicorn", "src.api.ml_service:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
