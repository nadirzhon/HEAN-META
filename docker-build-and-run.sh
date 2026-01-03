#!/bin/bash

# Скрипт для сборки и запуска HEAN через Docker

set -e

echo "🔨 Сборка Docker образа..."
docker build -t hean:latest .

echo ""
echo "✅ Образ собран успешно!"
echo ""
echo "🚀 Запуск через docker-compose..."
docker-compose up -d

echo ""
echo "✅ Контейнер запущен!"
echo ""
echo "📊 Просмотр логов:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 Остановка контейнера:"
echo "   docker-compose down"
echo ""
echo "📈 Статус контейнеров:"
docker-compose ps

