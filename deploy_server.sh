#!/bin/bash

echo "🌀 Pulling latest code..."
git pull origin main

echo "🧼 Stopping and removing old containers..."
docker-compose down

echo "🔨 Rebuilding images..."
docker-compose build

echo "🚀 Starting containers..."
docker-compose up -d

echo "🔍 Checking container status..."
docker-compose ps

echo "🎉 Deployment complete!"