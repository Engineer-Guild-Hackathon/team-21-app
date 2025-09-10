#!/usr/bin/env bash
set -euo pipefail

# Config (edit as needed)
PROJECT_ID=${PROJECT_ID:-web-summarizer-app}
REGION=${REGION:-asia-northeast1}
REPO=${REPO:-app-reg}
CLUSTER_NAME=${CLUSTER_NAME:-noncog-auto}
NAMESPACE=${NAMESPACE:-noncog}
BACKEND_IMAGE_TAG=${BACKEND_IMAGE_TAG:-$(git rev-parse --short HEAD)}
FRONTEND_IMAGE_TAG=${FRONTEND_IMAGE_TAG:-$(git rev-parse --short HEAD)}
DB_INSTANCE=${DB_INSTANCE:-noncog-pg}
STATIC_IP_NAME=${STATIC_IP_NAME:-noncog-ip}

echo "[1/8] gcloud project & region"
gcloud config set project "$PROJECT_ID" >/dev/null
gcloud config set compute/region "$REGION" >/dev/null

echo "[2/8] Artifact Registry setup"
gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1 || \
gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION"
gcloud auth configure-docker "$REGION"-docker.pkg.dev -q

BACKEND_IMG="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/backend:$BACKEND_IMAGE_TAG"
FRONTEND_IMG="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/frontend:$FRONTEND_IMAGE_TAG"

echo "[3/8] Build & Push images"
docker build -t "$BACKEND_IMG" backend
docker push "$BACKEND_IMG"
docker build -t "$FRONTEND_IMG" frontend
docker push "$FRONTEND_IMG"

echo "[4/8] GKE Autopilot cluster"
gcloud container clusters describe "$CLUSTER_NAME" --region "$REGION" >/dev/null 2>&1 || \
gcloud container clusters create-auto "$CLUSTER_NAME" --region "$REGION"
gcloud container clusters get-credentials "$CLUSTER_NAME" --region "$REGION"

echo "[5/8] Global Static IP"
gcloud compute addresses describe "$STATIC_IP_NAME" --global >/dev/null 2>&1 || \
gcloud compute addresses create "$STATIC_IP_NAME" --global
STATIC_IP=$(gcloud compute addresses describe "$STATIC_IP_NAME" --global --format='value(address)')
echo "STATIC_IP=$STATIC_IP"

echo "[6/8] Prepare k8s manifests (replace placeholders)"
WORKDIR=$(mktemp -d)
cp -r k8s "$WORKDIR/"
sed -i "s#__IMAGE_TAG__#$BACKEND_IMAGE_TAG#g" "$WORKDIR/k8s/backend.yaml"
sed -i "s#__IMAGE_TAG__#$FRONTEND_IMAGE_TAG#g" "$WORKDIR/k8s/frontend.yaml"
for f in "$WORKDIR/k8s/"*.yaml; do sed -i "s#__STATIC_IP__#$STATIC_IP#g" "$f"; done

echo "[7/8] Kubernetes apply"
kubectl apply -f "$WORKDIR/k8s/namespace.yaml"
kubectl -n "$NAMESPACE" apply -f "$WORKDIR/k8s/secrets.yaml"
kubectl -n "$NAMESPACE" apply -f "$WORKDIR/k8s/config.yaml"
kubectl -n "$NAMESPACE" apply -f "$WORKDIR/k8s/backend.yaml"
kubectl -n "$NAMESPACE" apply -f "$WORKDIR/k8s/frontend.yaml"
kubectl -n "$NAMESPACE" apply -f "$WORKDIR/k8s/ingress.yaml"

echo "[8/8] Done"
echo "API: https://api.$STATIC_IP.nip.io/docs"
echo "APP: https://app.$STATIC_IP.nip.io"

