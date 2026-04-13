#!/bin/bash
# ============================================================
# Reset HCA workspace — clears all generated projects and data
# ============================================================

set -e

echo "=== HCA Orchestration — Workspace Reset ==="
echo ""

read -p "This will delete ALL generated projects and database. Continue? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Clearing workspace..."
    rm -rf workspace/*
    touch workspace/.gitkeep

    echo "Clearing database..."
    rm -f data/hca.db data/hca.db-journal

    echo "Clearing Redis data..."
    docker compose exec redis redis-cli FLUSHALL 2>/dev/null || echo "(Redis not running, skipping)"

    echo ""
    echo "=== Workspace reset complete! ==="
else
    echo "Cancelled."
fi
