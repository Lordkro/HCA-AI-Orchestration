-- HCA v2 initial schema.
--
-- Projects and tasks are stored as the JSON form of the hca-core types
-- (`data`), plus the columns needed for querying. The state machines live
-- in hca-core; the database never changes a status on its own.
--
-- Timestamps are RFC 3339 UTC strings with microseconds and a `Z` suffix,
-- so they sort and compare correctly as text.

CREATE TABLE projects (
    id          TEXT PRIMARY KEY NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    data        TEXT NOT NULL
) STRICT;

CREATE INDEX idx_projects_status ON projects (status);

CREATE TABLE tasks (
    id          TEXT PRIMARY KEY NOT NULL,
    project_id  TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    state       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    data        TEXT NOT NULL
) STRICT;

CREATE INDEX idx_tasks_project ON tasks (project_id, created_at);
CREATE INDEX idx_tasks_state ON tasks (state);

-- Every accepted task event, in order. This is the project timeline.
CREATE TABLE task_events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    project_id  TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    from_state  TEXT NOT NULL,
    to_state    TEXT NOT NULL,
    event       TEXT NOT NULL,
    at          TEXT NOT NULL
) STRICT;

CREATE INDEX idx_task_events_task ON task_events (task_id, seq);
CREATE INDEX idx_task_events_project ON task_events (project_id, seq);

-- Durable work queue. Replaces Redis Streams.
--
-- A worker claims a job by taking a lease (`locked_by`, `locked_until`).
-- If the worker dies, the lease expires and another worker can claim it.
-- `attempts` counts claims; once it passes `max_attempts` the job is dead.
CREATE TABLE jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    task_id       TEXT REFERENCES tasks (id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,
    payload       TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'dead', 'cancelled')),
    attempts      INTEGER NOT NULL DEFAULT 0,
    max_attempts  INTEGER NOT NULL CHECK (max_attempts >= 1),
    run_after     TEXT NOT NULL,
    locked_by     TEXT,
    locked_until  TEXT,
    last_error    TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
) STRICT;

CREATE INDEX idx_jobs_claim ON jobs (status, run_after, id);
CREATE INDEX idx_jobs_project ON jobs (project_id, status);
