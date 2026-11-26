# Image for HCA project sandboxes: where agents run code and tests.
# Build: docker build -t hca-sandbox -f docker/sandbox.Dockerfile docker
#
# Containers from this image run with no network, so everything agents
# commonly need must be installed here.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git nodejs npm \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir pytest pytest-timeout requests fastapi httpx pydantic \
 && useradd --uid 1000 --create-home --shell /bin/sh agent \
 && useradd --uid 1001 --create-home --shell /bin/sh critic

USER 1000:1000
WORKDIR /workspace
CMD ["sleep", "infinity"]
