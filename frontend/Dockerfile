# 開発環境
FROM node:18-alpine AS development

WORKDIR /app

# 依存関係のインストール
COPY package*.json ./
RUN npm install
RUN npm install -D @tailwindcss/forms

# ソースコードのコピー
COPY . .

# 開発サーバーの起動
CMD ["npm", "run", "dev"]

# ビルド環境
FROM node:18-alpine AS builder

WORKDIR /app

# 依存関係のインストール
COPY package*.json ./
RUN npm install
RUN npm install -D @tailwindcss/forms

# ソースコードのコピー
COPY . .

# アプリケーションのビルド
RUN npm run build

# 本番環境
FROM node:18-alpine AS production

WORKDIR /app

# 本番環境の依存関係のみをインストール
COPY package*.json ./
RUN npm ci --only=production
RUN npm install -D @tailwindcss/forms

# ビルド成果物のコピー
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json

# 本番サーバーの起動
CMD ["npm", "start"]