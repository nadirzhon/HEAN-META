#!/bin/bash

# Script to build and run the website in Docker

echo "🚀 Building HEAN website Docker image..."
docker build -t hean-website .

echo "📦 Starting website container..."
docker run -d \
  --name hean-website \
  -p 3000:80 \
  --restart unless-stopped \
  hean-website

echo "✅ Website is running at http://localhost:3000"
echo "📊 Check status: docker ps | grep hean-website"
echo "📝 View logs: docker logs -f hean-website"
echo "🛑 Stop: docker stop hean-website"
echo "🗑️  Remove: docker rm -f hean-website"

