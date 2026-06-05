# Workspace Management

The HCA Orchestration system automatically manages project workspaces to prevent exponential disk usage growth.

## Overview

Each project generates a unique workspace directory (UUID-named) under `.data/workspaces/` containing:
- `src/` — Generated source code
- `tests/` — Generated test files

Without retention policies, these would accumulate indefinitely. The system now enforces configurable limits.

## Configuration

Edit environment variables to customize retention behavior:

```bash
# Maximum age for workspaces (days)
# Default: 7 days
WORKSPACE_RETENTION_DAYS=7

# Maximum number of workspaces to keep
# Keeps the N most recent workspaces
# Default: 100 workspaces
WORKSPACE_MAX_COUNT=100
```

## Automatic Cleanup

Cleanup runs automatically every hour during normal operation. The pipeline monitor:

1. Checks workspace ages against `WORKSPACE_RETENTION_DAYS`
2. Trims to keep only the `WORKSPACE_MAX_COUNT` newest workspaces
3. Logs metrics: cleaned count, size freed, remaining workspaces

Example log output:
```json
{"event": "workspace_cleanup_complete", "cleaned": 2, "size_freed_mb": 145, "total_workspaces": 13, "remaining": 11, "level": "info"}
```

## Manual Operations

### Get Workspace Stats
```bash
curl http://localhost:8080/api/projects/workspaces/stats
```

Response:
```json
{
  "total_count": 13,
  "total_size_mb": 340,
  "avg_size_mb": 26,
  "oldest_age_days": 45,
  "newest_age_days": 0
}
```

### Trigger Cleanup
```bash
curl -X POST http://localhost:8080/api/projects/workspaces/cleanup
```

Response:
```json
{
  "cleaned": 3,
  "size_freed_mb": 87,
  "total_workspaces": 13,
  "remaining": 10
}
```

## Directory Structure

```
.data/
├── workspaces/          # Project workspaces (auto-cleaned)
│   ├── <uuid-1>/
│   │   ├── src/
│   │   └── tests/
│   ├── <uuid-2>/
│   │   ├── src/
│   │   └── tests/
│   └── ...
├── logs/                # Application logs
└── cache/               # Temporary cache
```

## Retention Policy Example

With default settings (`WORKSPACE_RETENTION_DAYS=7`, `WORKSPACE_MAX_COUNT=100`):

- **Day 0**: Project 1 workspace created (~30MB)
- **Day 1**: Project 2 workspace created (~25MB)
- **Day 7**: Project 8 workspace created; cleanup runs at 1 hour mark
  - Project 1 workspace (7+ days old) → **removed**
  - Freed: ~30MB
- **Day 15**: 8 workspaces exist; cleanup keeps newest 8
  - If projects 9-100 created before day 15, oldest workspaces trimmed
  - Keep max 100, newest ones always preserved

## Monitoring

Check application logs for workspace cleanup events:

```bash
# Show workspace cleanup logs
docker compose logs hca-orchestrator | grep workspace_cleanup
```

Or programmatically:
```python
from hca.orchestrator.workspace_manager import WorkspaceManager

# Get stats
stats = await WorkspaceManager.get_workspace_stats()
print(f"Total: {stats['total_count']} workspaces, {stats['total_size_mb']}MB")

# Run cleanup
result = await WorkspaceManager.cleanup_old_workspaces()
print(f"Cleaned {result['cleaned']} workspaces, freed {result['size_freed_mb']}MB")
```

## Performance Impact

- **Cleanup overhead**: ~100ms per 100 workspaces
- **Scheduled**: Every 1 hour (configurable in `src/hca/orchestrator/pipeline.py`)
- **Non-blocking**: Cleanup runs async during health checks

No impact on project execution or API responsiveness.
