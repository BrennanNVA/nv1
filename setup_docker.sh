#!/bin/bash
# Setup script for Docker permissions and TimescaleDB

set -e

echo "=== Nova Aetus Docker Setup ==="
echo ""

# Check if user is in docker group
if groups | grep -q docker; then
    echo "✓ User is in docker group"
else
    echo "⚠️  User is NOT in docker group"
    echo ""
    echo "To fix Docker permissions, run:"
    echo "  sudo usermod -aG docker \$USER"
    echo "  newgrp docker"
    echo ""
    echo "Or use sudo for docker commands (not recommended for production)"
fi

echo ""
echo "Starting TimescaleDB with docker-compose..."
docker-compose up -d

echo ""
echo "Waiting for database to be ready..."
sleep 5

echo ""
echo "Checking database status..."
docker-compose ps

echo ""
echo "=== Database Setup Complete ==="
echo ""
echo "To initialize the schema, run:"
echo "  source venv/bin/activate"
echo "  python3 -c \"import asyncio; from src.nova.data.storage import StorageService; from src.nova.core.config import load_config; asyncio.run((lambda: StorageService(load_config().data).connect().__anext__())())\""
echo ""
echo "Or use the main application which will auto-initialize on first run."
