# 開発環境
FROM node:18-alpine AS development

WORKDIR /app

# 依存関係のインストール
COPY package*.json ./
RUN npm install

# ソースコードのコピー
COPY . .

# 開発サーバーの起動
CMD ["npm", "run", "dev"]

# ビルド環境
FROM node:18-alpine AS builder

WORKDIR /app

# 依存関係のインストール（開発依存関係を含む）
COPY package*.json ./
RUN npm ci

# ソースコードのコピー
COPY . .

# 設定ファイルのコピー
COPY next.config.js ./
COPY tsconfig.json ./
COPY postcss.config.js ./
COPY tailwind.config.ts ./

# アプリケーションのビルド
RUN npm run build

# 本番環境
FROM node:18-alpine AS production

WORKDIR /app

# package.jsonとpackage-lock.jsonをコピー
COPY package*.json ./

# 本番環境の依存関係をインストール
RUN npm ci --omit=dev

# ビルド成果物のコピー
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.js ./
COPY --from=builder /app/package.json ./

# 本番サーバーの起動
CMD ["npm", "start"]