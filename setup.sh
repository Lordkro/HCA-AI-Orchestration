#!/usr/bin/env bash
# ============================================================
# HCA Orchestration — One-command Setup Script
# ============================================================
# Detects your hardware, configures .env, starts services,
# and pulls the required LLM models — all in one step.
#
# Usage:  bash setup.sh
#         bash setup.sh --profile nvidia   # force NVIDIA profile
#         bash setup.sh --profile rocm     # force AMD ROCm profile
#         bash setup.sh --profile vulkan   # force Vulkan GPU profile
#         bash setup.sh --profile metal    # force Apple Silicon (Metal) profile
#         bash setup.sh --profile cpu      # force CPU-only mode
#         bash setup.sh --models "llama3.2:3b qwen2.5-coder:3b"
#         bash setup.sh --skip-pull        # skip model pulling
#         bash setup.sh --help             # show full usage
# ============================================================

set -euo pipefail

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
info()  { echo -e "${CYAN}INFO${NC}  $*"; }
ok()    { echo -e "${GREEN}OK${NC}    $*"; }
warn()  { echo -e "${YELLOW}WARN${NC}  $*"; }
fail()  { echo -e "${RED}FAIL${NC}  $*"; }
header(){
    local text="$*"
    local line=""
    for ((i=0; i<${#text}; i++)); do line+="="; done
    echo -e "\n${BOLD}${text}${NC}\n${line}\n"
}

# ──────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────
PROFILE="auto"
MODELS_TO_PULL=""
SKIP_PULL=false

usage() {
    cat <<EOF
Usage: bash setup.sh [OPTIONS]

Options:
  --profile <profile>    Force a Docker Compose profile:
                         auto | cpu | nvidia | rocm | vulkan | metal  (default: auto)
  --models "<list>"      Models to pull, space-separated
                         (default: qwen3:14b qwen2.5-coder:14b)
  --skip-pull            Skip pulling LLM models
  --help                 Show this help and exit

Examples:
  bash setup.sh                          # Auto-detect + full setup
  bash setup.sh --profile nvidia         # Force NVIDIA profile
  bash setup.sh --profile vulkan         # Force Vulkan GPU profile
  bash setup.sh --profile metal          # Force Apple Silicon (Metal) profile
  bash setup.sh --models "llama3.2:3b qwen2.5-coder:3b"
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)     PROFILE="$2"; shift 2 ;;
        --models)      MODELS_TO_PULL="$2"; shift 2 ;;
        --skip-pull)   SKIP_PULL=true; shift ;;
        --help)        usage ;;
        *)             echo "Unknown option: $1"; usage ;;
    esac
done

# ──────────────────────────────────────────────
# 1. Prerequisite checks
# ──────────────────────────────────────────────
header "[1/7] Checking prerequisites"

# Docker
if command -v docker &>/dev/null; then
    ok "Docker found: $(docker --version)"
else
    fail "Docker is not installed."
    echo "  Install Docker: https://docs.docker.com/engine/install/"
    exit 1
fi

# Docker Compose (v2 plugin or standalone)
COMPOSE_CMD=""
if docker compose version &>/dev/null; then
    COMPOSE_CMD="docker compose"
    ok "Docker Compose found: $(docker compose version 2>&1)"
elif command -v docker-compose &>/dev/null; then
    COMPOSE_CMD="docker-compose"
    ok "Docker Compose found: $(docker-compose --version 2>&1)"
else
    fail "Docker Compose is not installed."
    echo "  Install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Git
if command -v git &>/dev/null; then
    ok "Git found: $(git --version)"
else
    warn "Git not found — workspace versioning will be unavailable."
fi

# curl (for model puller fallback)
if ! command -v curl &>/dev/null; then
    fail "curl is required. Install it and re-run."
    exit 1
fi

# ──────────────────────────────────────────────
# 2. Hardware detection & profile selection
# ──────────────────────────────────────────────
header "[2/7] Detecting hardware"

detect_gpu_profile() {
    # Check for Apple Silicon (Metal)
    if [[ "$(uname -s)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        echo "metal"
        info "Apple Silicon (M-series) detected — using Metal GPU profile."
        return
    fi

    # Check for NVIDIA GPU
    if command -v nvidia-smi &>/dev/null; then
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
        if [[ -n "$gpu_info" ]]; then
            echo "nvidia"
            info "NVIDIA GPU detected: $gpu_info"
            # Check nvidia-container-toolkit
            if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
                warn "nvidia-container-toolkit not detected in Docker."
                echo "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
                echo "  Then restart Docker and re-run setup."
            fi
            return
        fi
    fi

    # Check for AMD ROCm
    if [[ -e /dev/kfd ]] && [[ -e /dev/dri ]]; then
        echo "rocm"
        info "AMD ROCm device detected"
        return
    fi

    # Check for Vulkan-capable GPU (any GPU with /dev/dri render node)
    if [[ -e /dev/dri/renderD128 ]]; then
        echo "vulkan"
        info "GPU detected — using Vulkan profile (recommended fallback for most GPUs)."
        return
    fi

    # CPU-only fallback
    echo "cpu"
    info "No supported GPU detected — using CPU-only profile."
}

if [[ "$PROFILE" == "auto" ]]; then
    PROFILE=$(detect_gpu_profile)
    ok "Selected profile: ${PROFILE}"
else
    ok "Using user-specified profile: ${PROFILE}"
fi

case "$PROFILE" in
    cpu)    PROFILE_FLAG=""                     ; OLLAMA_SERVICE="ollama" ;;
    nvidia) PROFILE_FLAG="--profile nvidia"     ; OLLAMA_SERVICE="ollama-nvidia" ;;
    rocm)   PROFILE_FLAG="--profile rocm"       ; OLLAMA_SERVICE="ollama-rocm" ;;
    vulkan) PROFILE_FLAG="--profile vulkan"     ; OLLAMA_SERVICE="ollama-vulkan" ;;
    metal)  PROFILE_FLAG="--profile metal"      ; OLLAMA_SERVICE="ollama-metal" ;;
    *)
        fail "Unknown profile: $PROFILE. Valid: cpu, nvidia, rocm, vulkan, metal"
        exit 1
        ;;
esac

# ──────────────────────────────────────────────
# 3. Environment configuration
# ──────────────────────────────────────────────
header "[3/7] Configuring environment"

ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"

if [[ -f "$ENV_FILE" ]]; then
    ok ".env file already exists — keeping existing configuration."
else
    if [[ -f "$ENV_EXAMPLE" ]]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        ok "Created .env from .env.example"
    else
        warn ".env.example not found — creating minimal .env"
        cat > "$ENV_FILE" <<-EOF
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_DEFAULT_MODEL=qwen3:14b
OLLAMA_CODER_MODEL=qwen2.5-coder:14b
REDIS_URL=redis://redis:6379/0
DATABASE_URL=sqlite:///data/hca.db
WORKSPACE_DIR=/workspace
EOF
        ok "Created minimal .env"
    fi
fi

# ──────────────────────────────────────────────
# 4. Port conflict check
# ──────────────────────────────────────────────
header "[4/7] Checking port availability"

check_port() {
    local port=$1
    local service=$2
    if command -v ss &>/dev/null; then
        if ss -tlnp "sport = :$port" 2>/dev/null | grep -q ":$port"; then
            warn "Port $port ($service) is already in use."
            return 1
        fi
    elif command -v netstat &>/dev/null; then
        if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
            warn "Port $port ($service) is already in use."
            return 1
        fi
    fi
    return 0
}

PORTS_OK=true
check_port 11434 "Ollama" || PORTS_OK=false
check_port 6379  "Redis"  || PORTS_OK=false
check_port 8080  "Web UI" || PORTS_OK=false

if ! $PORTS_OK; then
    echo ""
    warn "Some ports are occupied. If running another Ollama/Redis instance,"
    echo "  stop it first or change ports in .env before proceeding."
    echo ""
    read -rp "Continue anyway? (y/N) " -n 1 reply
    echo ""
    if [[ ! "$reply" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# ──────────────────────────────────────────────
# 5. Ensure writable data directories
# ──────────────────────────────────────────────
header "[5/7] Ensuring writable data directories"

mkdir -p workspace .data
# Match the user:group set in docker-compose.yml (HOST_UID:HOST_GID)
chown "${HOST_UID:-1000}:${HOST_GID:-1000}" workspace .data 2>/dev/null || true
ok "Workspace directories ready"

# ──────────────────────────────────────────────
# 6. Start services
# ──────────────────────────────────────────────
header "[6/7] Starting services"

echo "Starting Ollama, Redis, and the Orchestrator..."
echo "  Profile: ${PROFILE}"
echo ""

# Pull images first (shows progress)
info "Pulling Docker images..."
if [[ -n "$PROFILE_FLAG" ]]; then
    $COMPOSE_CMD $PROFILE_FLAG pull "$OLLAMA_SERVICE" redis 2>&1 | tail -5 || true
else
    $COMPOSE_CMD pull "$OLLAMA_SERVICE" redis 2>&1 | tail -5 || true
fi

# Start services
info "Starting services..."
if [[ -n "$PROFILE_FLAG" ]]; then
    $COMPOSE_CMD $PROFILE_FLAG up -d "$OLLAMA_SERVICE" redis 2>&1 | tail -3
else
    $COMPOSE_CMD up -d "$OLLAMA_SERVICE" redis 2>&1 | tail -3
fi

# Wait for Ollama to be healthy
info "Waiting for Ollama to be ready..."
OLLAMA_READY=false
for i in $(seq 1 30); do
    if curl -sSf http://localhost:11434/api/tags &>/dev/null; then
        OLLAMA_READY=true
        ok "Ollama is ready"
        break
    fi
    sleep 2
done

if ! $OLLAMA_READY; then
    fail "Ollama did not become ready within 60 seconds."
    echo "  Check logs: docker compose logs ollama"
    exit 1
fi

# Start orchestrator (depends on Ollama + Redis being healthy)
info "Starting orchestrator..."
if [[ -n "$PROFILE_FLAG" ]]; then
    $COMPOSE_CMD $PROFILE_FLAG up -d orchestrator 2>&1 | tail -3
else
    $COMPOSE_CMD up -d orchestrator 2>&1 | tail -3
fi

# Wait for orchestrator health
info "Waiting for orchestrator..."
for i in $(seq 1 15); do
    if curl -sSf http://localhost:8080/health &>/dev/null; then
        ok "Orchestrator is ready at http://localhost:8080"
        break
    fi
    sleep 2
done

# ──────────────────────────────────────────────
# 6. Pull LLM models
# ──────────────────────────────────────────────
header "[7/7] Pulling LLM models"

if $SKIP_PULL; then
    info "Skipping model pull (--skip-pull)."
    echo "  Pull manually:  $COMPOSE_CMD exec ollama ollama pull <model>"
else
    if [[ -z "$MODELS_TO_PULL" ]]; then
        # Read from .env or use defaults
        # Pull models matching .env defaults (overridable via OLLAMA_MODELS_TO_PULL)
        MODELS_TO_PULL="${OLLAMA_MODELS_TO_PULL:-qwen3:8b qwen2.5-coder:7b}"
    fi

    echo "Models to pull: $MODELS_TO_PULL"
    echo ""

    for model in $MODELS_TO_PULL; do
        info "Pulling ${model}..."
        if curl -sS "http://localhost:11434/api/pull" \
                -d "{\"name\": \"$model\"}" \
                -o /dev/null -w "  HTTP status: %{http_code}\n"; then
            ok "Pulled ${model}"
        else
            warn "Failed to pull ${model}"
        fi
        echo ""
    done

    echo ""
    ok "Model pulling complete!"
    echo ""
    echo "Available models:"
    curl -s http://localhost:11434/api/tags | python3 -m json.tool 2>/dev/null \
        || curl -s http://localhost:11434/api/tags | head -c 200
fi

# ──────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────
header "Setup complete!"

echo -e "  ${GREEN}Dashboard:${NC}  http://localhost:8080"
echo -e "  ${GREEN}Ollama API:${NC} http://localhost:11434"
echo -e "  ${GREEN}Profile:${NC}   ${PROFILE}"
echo ""
echo "To stop all services:"
echo "  $COMPOSE_CMD${PROFILE_FLAG:+ $PROFILE_FLAG} down"
echo ""
echo "To view logs:"
echo "  $COMPOSE_CMD${PROFILE_FLAG:+ $PROFILE_FLAG} logs -f"
echo ""
echo "To submit your first project, open the dashboard and enter a product idea!"
echo ""
