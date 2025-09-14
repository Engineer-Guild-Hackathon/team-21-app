#!/bin/bash

# Kafkaヘルスチェックスクリプト
# Kafkaクラスターの状態とトピックの健全性をチェック

set -e

KAFKA_BOOTSTRAP_SERVER="kafka:9092"

echo "🔍 Kafka Health Check for NonCog Learning Platform"
echo "=================================================="

# 1. Kafka接続テスト
echo "1️⃣ Testing Kafka connection..."
if docker-compose exec kafka kafka-topics --list --bootstrap-server "$KAFKA_BOOTSTRAP_SERVER" > /dev/null 2>&1; then
    echo "✅ Kafka connection successful"
else
    echo "❌ Kafka connection failed"
    exit 1
fi

# 2. トピック一覧の表示
echo ""
echo "2️⃣ Available topics:"
docker-compose exec kafka kafka-topics --list --bootstrap-server "$KAFKA_BOOTSTRAP_SERVER"

# 3. 主要トピックの詳細情報
echo ""
echo "3️⃣ Topic details:"
TOPICS=("learn_action_events" "student_activity" "learning_metrics")

for topic in "${TOPICS[@]}"; do
    echo "📊 Topic: $topic"
    docker-compose exec kafka kafka-topics \
        --describe \
        --topic "$topic" \
        --bootstrap-server "$KAFKA_BOOTSTRAP_SERVER" 2>/dev/null || echo "  ⚠️  Topic not found"
    echo ""
done

# 4. コンシューマーグループの確認
echo "4️⃣ Consumer groups:"
docker-compose exec kafka kafka-consumer-groups \
    --list \
    --bootstrap-server "$KAFKA_BOOTSTRAP_SERVER" 2>/dev/null || echo "  No consumer groups found"

echo ""
echo "🎯 Kafka Health Check completed!"
echo "💡 To monitor real-time messages, use:"
echo "   docker-compose exec kafka kafka-console-consumer --topic learn_action_events --bootstrap-server $KAFKA_BOOTSTRAP_SERVER --from-beginning"
