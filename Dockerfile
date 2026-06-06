FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd --system --gid 1001 hca && \
    useradd --system --gid hca --uid 1001 --create-home --shell /sbin/nologin hca

# Copy dependency definition first for layer caching
COPY pyproject.toml README.md ./

# Install Python dependencies (cached as long as pyproject.toml doesn't change)
RUN pip install --no-cache-dir .

# Copy application source and scripts
COPY src/ src/
COPY scripts/ scripts/

# Create data directories and set ownership
RUN mkdir -p .data/workspaces .data/logs .data/cache && \
    chown -R hca:hca .data

# Switch to non-root user
USER hca:hca

# Expose the web UI port
EXPOSE 8080

# Run the orchestrator
CMD ["python", "-m", "hca.main"]
