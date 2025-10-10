#!/bin/bash
set -e

echo "🔄 Waiting for MySQL to be ready..."
while ! mysqladmin ping -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASSWORD" --silent; do
    echo "Waiting for MySQL..."
    sleep 2
done

echo "✅ MySQL is ready!"

echo "🔄 Seeding database..."
python scripts/seed_database.py

echo "🔄 Running pipeline..."
python pipeline.py

echo "✅ Pipeline completed!"
