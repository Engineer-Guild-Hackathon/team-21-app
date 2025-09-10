# フロントエンド用Dockerfile
FROM node:18-alpine AS base

# 依存関係のインストール
FROM base AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# 開発環境用ビルド
FROM base AS development
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev"]

# 本番環境用ビルド
FROM base AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM base AS production
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY package*.json ./
EXPOSE 3000
CMD ["npm", "start"]