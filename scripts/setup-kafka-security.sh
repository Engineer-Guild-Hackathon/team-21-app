#!/bin/bash

# Kafkaセキュリティ設定スクリプト
# 本番環境用のセキュリティ機能を有効化

set -e

echo "🔒 Setting up Kafka Security for Production Environment"
echo "====================================================="

# 環境変数の確認
if [ -z "$KAFKA_SSL_PASSWORD" ]; then
    echo "❌ KAFKA_SSL_PASSWORD environment variable is required"
    echo "   export KAFKA_SSL_PASSWORD=your-secure-password"
    exit 1
fi

# SSL証明書ディレクトリの作成
echo "📁 Creating SSL certificate directories..."
mkdir -p ssl
mkdir -p kafka-users

# 1. SSL証明書の生成
echo "🔐 Generating SSL certificates..."

# CA証明書の生成
openssl req -new -x509 -keyout ssl/ca-key -out ssl/ca-cert -days 365 -subj "/C=JP/ST=Tokyo/L=Tokyo/O=NonCog/CN=ca" -passout pass:$KAFKA_SSL_PASSWORD

# キーストアの生成
keytool -keystore ssl/kafka.server.keystore.jks -alias kafka -validity 365 -genkey -keyalg RSA -dname "CN=kafka,OU=NonCog,O=NonCog,L=Tokyo,S=Tokyo,C=JP" -storepass $KAFKA_SSL_PASSWORD -keypass $KAFKA_SSL_PASSWORD

# CA証明書のインポート
keytool -keystore ssl/kafka.server.truststore.jks -alias CARoot -import -file ssl/ca-cert -storepass $KAFKA_SSL_PASSWORD -noprompt

# 証明書署名要求の生成
keytool -keystore ssl/kafka.server.keystore.jks -alias kafka -certreq -file ssl/cert-file -storepass $KAFKA_SSL_PASSWORD

# 証明書の署名
openssl x509 -req -CA ssl/ca-cert -CAkey ssl/ca-key -in ssl/cert-file -out ssl/cert-signed -days 365 -CAcreateserial -passin pass:$KAFKA_SSL_PASSWORD

# 署名された証明書のインポート
keytool -keystore ssl/kafka.server.keystore.jks -alias CARoot -import -file ssl/ca-cert -storepass $KAFKA_SSL_PASSWORD -noprompt
keytool -keystore ssl/kafka.server.keystore.jks -alias kafka -import -file ssl/cert-signed -storepass $KAFKA_SSL_PASSWORD -noprompt

# 2. JAAS設定ファイルの作成
echo "👤 Creating JAAS configuration files..."

# Kafka Server JAAS設定
cat > kafka-server_jaas.conf << EOF
KafkaServer {
    org.apache.kafka.common.security.plain.PlainLoginModule required
    username="admin"
    password="admin-secret"
    user_admin="admin-secret"
    user_learning-app="app-secret"
    user_monitoring="monitor-secret";
};

Client {
    org.apache.kafka.common.security.plain.PlainLoginModule required
    username="admin"
    password="admin-secret";
};
EOF

# Zookeeper JAAS設定
cat > zookeeper_jaas.conf << EOF
Server {
    org.apache.kafka.common.security.plain.PlainLoginModule required
    username="admin"
    password="admin-secret"
    user_admin="admin-secret";
};
EOF

# 3. ユーザー設定ファイルの作成
echo "👥 Creating user configuration files..."

cat > kafka-users/users.properties << EOF
# Kafka users configuration
# Format: username=password

admin=admin-secret
learning-app=app-secret
monitoring=monitor-secret
EOF

# 4. 環境変数ファイルの作成
echo "🌍 Creating environment configuration..."

cat > .env.prod << EOF
# Kafka Production Environment Variables
KAFKA_SSL_PASSWORD=$KAFKA_SSL_PASSWORD
KAFKA_ADMIN_USERNAME=admin
KAFKA_ADMIN_PASSWORD=admin-secret
KAFKA_APP_USERNAME=learning-app
KAFKA_APP_PASSWORD=app-secret
KAFKA_MONITOR_USERNAME=monitoring
KAFKA_MONITOR_PASSWORD=monitor-secret
EOF

echo ""
echo "✅ Kafka Security Setup Completed!"
echo ""
echo "📋 Generated Files:"
echo "  - ssl/ca-cert (CA Certificate)"
echo "  - ssl/kafka.server.keystore.jks (Server Keystore)"
echo "  - ssl/kafka.server.truststore.jks (Server Truststore)"
echo "  - kafka-server_jaas.conf (Kafka JAAS Config)"
echo "  - zookeeper_jaas.conf (Zookeeper JAAS Config)"
echo "  - kafka-users/users.properties (User Configuration)"
echo "  - .env.prod (Environment Variables)"
echo ""
echo "🚀 Next Steps:"
echo "  1. Review and customize the generated configuration files"
echo "  2. Update application code to use SASL_SSL authentication"
echo "  3. Test the secure configuration: docker-compose -f docker-compose.prod.yml up"
echo "  4. Set up monitoring for authentication failures"
echo ""
echo "⚠️  Security Notes:"
echo "  - Keep SSL passwords secure and rotate regularly"
echo "  - Monitor authentication logs for suspicious activity"
echo "  - Regularly update SSL certificates before expiration"
echo "  - Use strong passwords for all user accounts"
