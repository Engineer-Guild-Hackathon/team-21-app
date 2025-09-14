#!/bin/bash

# Kafkaトピック初期化スクリプト
# 学習プラットフォーム用のKafkaトピックを作成

set -e

KAFKA_BOOTSTRAP_SERVER="kafka:9092"
TOPICS=(
    "learn_action_events:3:1"
    "student_activity:3:1" 
    "learning_metrics:3:1"
    "emotional_state:3:1"
    "feedback_events:3:1"
    "ml_analysis_requests:3:1"
    "ml_analysis_results:3:1"
)

echo "🚀 Initializing Kafka topics for NonCog Learning Platform..."

for topic_config in "${TOPICS[@]}"; do
    IFS=':' read -r topic_name partitions replication <<< "$topic_config"
    
    echo "📝 Creating topic: $topic_name (partitions: $partitions, replication: $replication)"
    
    docker-compose exec -T kafka kafka-topics \
        --create \
        --topic "$topic_name" \
        --bootstrap-server "$KAFKA_BOOTSTRAP_SERVER" \
        --partitions "$partitions" \
        --replication-factor "$replication" \
        --if-not-exists || echo "⚠️  Topic $topic_name may already exist"
done

echo "✅ Kafka topics initialization completed!"

echo "📋 Listing all topics:"
docker-compose exec kafka kafka-topics --list --bootstrap-server "$KAFKA_BOOTSTRAP_SERVER"

echo ""
echo "🎯 Topics created for NonCog Learning Platform:"
echo "  - learn_action_events: 学習行動イベント"
echo "  - student_activity: 生徒アクティビティ"
echo "  - learning_metrics: 学習メトリクス"
echo "  - emotional_state: 感情状態データ"
echo "  - feedback_events: フィードバックイベント"
echo "  - ml_analysis_requests: ML分析リクエスト"
echo "  - ml_analysis_results: ML分析結果"
