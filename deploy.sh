#!/bin/bash
# deploy.sh — Build, push to ECR, and restart the container on EC2
# Usage: ./deploy.sh
# Requirements: AWS CLI configured, Docker running, SSH key available

set -e  # exit on any error

# ── Config ────────────────────────────────────────────────────────────────────
AWS_REGION="eu-north-1"
AWS_ACCOUNT_ID="557170680973"
ECR_REPO="rag-indian-law"
IMAGE_TAG="latest"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

EC2_USER="ubuntu"
EC2_HOST="54.209.6.232"
EC2_SSH_KEY=".key1.pem"   

CONTAINER_NAME="rag-indian-law"
S3_BUCKET="rag-indian-law-index"
EMBED_MODEL="law-ai/InLegalBERT"

# Load OPENAI_API_KEY from local .env if not already set
if [ -z "$OPENAI_API_KEY" ] && [ -f ".env" ]; then
    export $(grep OPENAI_API_KEY .env | xargs)
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "[error] OPENAI_API_KEY is not set. Add it to .env or export it before running."
    exit 1
fi

# ── Step 1: Build ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/4] Building Docker image..."
docker build --no-cache -t ${ECR_REPO} .

# ── Step 2: Push to ECR ───────────────────────────────────────────────────────
echo ""
echo ">>> [2/4] Pushing to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

docker tag ${ECR_REPO}:${IMAGE_TAG} ${ECR_URI}
docker push ${ECR_URI}

# ── Step 3: Deploy on EC2 ─────────────────────────────────────────────────────
echo ""
echo ">>> [3/4] Deploying on EC2..."
ssh -i ${EC2_SSH_KEY} -o StrictHostKeyChecking=no ${EC2_USER}@${EC2_HOST} << EOF
    set -e

    # Login to ECR
    aws ecr get-login-password --region ${AWS_REGION} | \
        docker login --username AWS --password-stdin \
        ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

    # Pull latest image
    docker pull ${ECR_URI}

    # Stop and remove old container (ignore error if not running)
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true

    # Run new container
    docker run -d \
        -p 8000:8000 \
        --name ${CONTAINER_NAME} \
        --restart unless-stopped \
        -e OPENAI_API_KEY=${OPENAI_API_KEY} \
        -e S3_INDEX_BUCKET=${S3_BUCKET} \
        -e S3_INDEX_PREFIX=index \
        -e AWS_DEFAULT_REGION=${AWS_REGION} \
        -e EMBED_MODEL=${EMBED_MODEL} \
        ${ECR_URI}

    # Clean up old images to free disk space
    docker image prune -f
EOF

# ── Step 4: Health check ──────────────────────────────────────────────────────
echo ""
echo ">>> [4/4] Waiting for app to start..."
sleep 15
curl -sf http://${EC2_HOST}:8000/health && echo "" && echo "Deployment successful! App is live at https://shastrashaw.online" \
    || echo "[warn] Health check failed — run: ssh -i ${EC2_SSH_KEY} ${EC2_USER}@${EC2_HOST} 'docker logs ${CONTAINER_NAME}'"
