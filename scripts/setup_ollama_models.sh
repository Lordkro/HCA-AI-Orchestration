#!/bin/bash
# ============================================================
# Pull required Ollama models
# ============================================================
# Run this after starting the Ollama container:
#   docker compose run --rm model-puller
#
# Or manually:
#   docker compose exec ollama ollama pull qwen2.5:32b
#   docker compose exec ollama ollama pull qwen2.5-coder:32b
# ============================================================

set -e

echo "=== HCA Orchestration — Model Setup ==="
echo ""

OLLAMA_HOST="${OLLAMA_BASE_URL:-http://localhost:11434}"

echo "Pulling qwen2.5:32b (default model)..."
curl -s "$OLLAMA_HOST/api/pull" -d '{"name": "qwen2.5:32b"}' | tail -1
echo ""

echo "Pulling qwen2.5-coder:32b (coder model)..."
curl -s "$OLLAMA_HOST/api/pull" -d '{"name": "qwen2.5-coder:32b"}' | tail -1
echo ""

echo "=== All models pulled successfully! ==="
echo ""
echo "Available models:"
curl -s "$OLLAMA_HOST/api/tags" | python3 -m json.tool 2>/dev/null || echo "(install python3 for formatted output)"
