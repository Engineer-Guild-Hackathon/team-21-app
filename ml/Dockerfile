# 開発環境
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime AS development

WORKDIR /app

# システムの依存関係
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# ソースコードのコピー
COPY . .

# 開発サーバーの起動
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

# 本番環境
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime AS production

WORKDIR /app

# システムの依存関係
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# ソースコードのコピー
COPY . .

# 本番サーバーの起動
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
