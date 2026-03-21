#!/bin/bash
set -euo pipefail

echo "Starting deployment..."
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ ! -f .env ]; then
  echo "Error: .env file not found"
  exit 1
fi

source .env
docker-compose build
docker-compose up -d
echo "Deployment complete"
