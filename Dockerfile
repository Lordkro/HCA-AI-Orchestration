FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create data and workspace directories
RUN mkdir -p /data /workspace

# Expose the web UI port
EXPOSE 8080

# Run the orchestrator
CMD ["python", "-m", "src.main"]
