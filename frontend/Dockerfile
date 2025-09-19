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

# 設定ファイルのコピー
COPY next.config.js ./
COPY tsconfig.json ./
COPY postcss.config.js ./
COPY tailwind.config.ts ./

# publicディレクトリの作成と静的ファイルのコピー
RUN mkdir -p public
COPY public ./public

# srcディレクトリのコピー
COPY src ./src

# アプリケーションのビルド
ENV NEXT_TELEMETRY_DISABLED 1
ARG NEXT_PUBLIC_API_BASE_URL=https://api.34.107.156.246.nip.io
ARG NEXT_PUBLIC_GEMINI_API_KEY=
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_GEMINI_API_KEY=$NEXT_PUBLIC_GEMINI_API_KEY
RUN npm run build

# 本番環境
FROM node:18-alpine AS production

WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

# package.jsonとpackage-lock.jsonをコピー
COPY package*.json ./

# 本番環境の依存関係をインストール
RUN npm ci --omit=dev

# 設定ファイルをコピー
COPY --from=builder /app/next.config.js ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# 本番サーバーの起動
CMD ["node", "server.js"]