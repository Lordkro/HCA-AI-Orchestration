-- What the team did, in order: plans, assignments, deliverables, reviews,
-- project status changes. Shown as the project's conversation in the
-- dashboard, and used to pass a task's latest deliverable to its reviewer.
CREATE TABLE activity (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    task_id     TEXT REFERENCES tasks (id) ON DELETE CASCADE,
    sender      TEXT NOT NULL,
    recipient   TEXT NOT NULL,
    kind        TEXT NOT NULL,
    content     TEXT NOT NULL,
    at          TEXT NOT NULL
) STRICT;

CREATE INDEX idx_activity_project ON activity (project_id, seq);
CREATE INDEX idx_activity_task ON activity (task_id, kind, seq);
