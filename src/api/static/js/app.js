// ============================================================
// HCA Orchestration — Dashboard Frontend
// ============================================================

const API_BASE = window.location.origin;
let ws = null;
let currentProjectId = null;
let currentProjectStatus = null;

const ROLE_ICONS = {
    pm: '📋', research: '🔍', spec: '📐', coder: '💻',
    critic: '🔎', user: '👤', system: '⚙️',
};

const TYPE_LABELS = {
    task_assignment: 'assign', deliverable: 'deliverable', feedback: 'feedback',
    status_update: 'status', question: 'question', answer: 'answer', system: 'system',
};

const FILE_ICONS = {
    py: '🐍', js: '📜', ts: '📘', html: '🌐', css: '🎨',
    json: '📋', yaml: '📋', yml: '📋', toml: '📋', md: '📝',
    txt: '📄', sh: '🐚', sql: '🗃️', dockerfile: '🐳',
};

// ============================================================
// View / Tab Navigation
// ============================================================

function switchView(view) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));

    document.getElementById(`view${capitalize(view)}`).classList.add('active');
    document.querySelector(`.tab[data-view="${view}"]`).classList.add('active');

    if (view === 'project' && currentProjectId) {
        loadProjectDetail(currentProjectId);
    }
}

function switchSubTab(tab) {
    document.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.subtab-content').forEach(c => c.classList.remove('active'));

    document.querySelector(`.sub-tab[data-subtab="${tab}"]`).classList.add('active');
    document.getElementById(`subtab${capitalize(tab)}`).classList.add('active');
}

function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
}

// ============================================================
// WebSocket Connection
// ============================================================

function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting in 3s...');
        updateConnectionStatus(false);
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => console.error('WebSocket error:', err);

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleRealtimeEvent(data);
        } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
        }
    };
}

function updateConnectionStatus(connected) {
    const el = document.getElementById('connectionStatus');
    el.innerHTML = connected
        ? '<span class="status-dot connected"></span><span>Connected</span>'
        : '<span class="status-dot disconnected"></span><span>Disconnected</span>';
}

function handleRealtimeEvent(data) {
    // Agent message → activity feed
    if (data.sender && data.payload) {
        addActivityItem(data);
    }

    // UI events (task_state_changed, agent_heartbeat, project_status_changed)
    if (data.type === 'agent_heartbeat') {
        loadAgents();
    }

    if (data.type === 'task_state_changed' || data.type === 'project_status_changed') {
        // Refresh project detail if we're viewing it
        if (currentProjectId && data.data) {
            const eventData = typeof data.data === 'string' ? JSON.parse(data.data) : data.data;
            if (eventData.project_id === currentProjectId) {
                loadProjectDetail(currentProjectId);
            }
        }
        loadProjects();
    }
}

// ============================================================
// API Calls
// ============================================================

async function apiFetch(path, opts = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
        headers: { 'Content-Type': 'application/json' },
        ...opts,
    });
    if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(err.detail || `HTTP ${response.status}`);
    }
    return response.json();
}

async function submitIdea() {
    const ideaInput = document.getElementById('ideaInput');
    const nameInput = document.getElementById('projectName');
    const idea = ideaInput.value.trim();

    if (!idea) { alert('Please enter a product idea.'); return; }

    const btn = document.getElementById('submitIdea');
    btn.disabled = true;
    btn.textContent = 'Submitting...';

    try {
        const result = await apiFetch('/api/projects/', {
            method: 'POST',
            body: JSON.stringify({ idea, name: nameInput.value.trim() || '' }),
        });

        ideaInput.value = '';
        nameInput.value = '';

        addActivityItem({
            sender: 'user', recipient: 'pm', type: 'system',
            payload: { content: `New project submitted: "${idea.substring(0, 100)}..."` },
            timestamp: new Date().toISOString(),
        });

        loadProjects();

        // Auto-navigate to the new project
        viewProject(result.project_id);
    } catch (err) {
        console.error('Failed to submit idea:', err);
        alert('Failed to submit idea. Is the server running?');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Submit to Team';
    }
}

async function loadAgents() {
    try {
        const agents = await apiFetch('/api/agents/');
        renderAgents(agents);
    } catch (err) {
        console.error('Failed to load agents:', err);
    }
}

async function loadProjects() {
    try {
        const projects = await apiFetch('/api/projects/');
        renderProjects(projects);
    } catch (err) {
        console.error('Failed to load projects:', err);
    }
}

async function loadProjectDetail(projectId) {
    try {
        const data = await apiFetch(`/api/projects/${projectId}`);
        renderProjectDetail(data);

        // Load sub-tabs
        const [tasks, messages, artifacts] = await Promise.all([
            apiFetch(`/api/tasks/${projectId}`),
            apiFetch(`/api/projects/${projectId}/messages`),
            apiFetch(`/api/projects/${projectId}/artifacts`),
        ]);

        renderKanban(tasks);
        renderConversation(messages);
        renderArtifacts(artifacts);
    } catch (err) {
        console.error('Failed to load project detail:', err);
    }
}

async function togglePause() {
    if (!currentProjectId) return;

    const action = currentProjectStatus === 'active' ? 'pause' : 'resume';
    try {
        await apiFetch(`/api/projects/${currentProjectId}/${action}`, { method: 'POST' });
        await loadProjectDetail(currentProjectId);
        loadProjects();
    } catch (err) {
        alert(`Failed to ${action} project: ${err.message}`);
    }
}

async function retryTask(taskId) {
    try {
        await apiFetch(`/api/tasks/detail/${taskId}/retry`, { method: 'POST' });
        closeModal('taskModal');
        if (currentProjectId) loadProjectDetail(currentProjectId);
    } catch (err) {
        alert(`Failed to retry task: ${err.message}`);
    }
}

async function loadHealth() {
    try {
        const health = await apiFetch('/api/health');
        const el = document.getElementById('healthStats');
        const published = health.bus?.messages_published || 0;
        const tokens = health.ollama?.total_tokens_used || 0;
        el.textContent = `msgs: ${published} | tokens: ${formatNumber(tokens)}`;
    } catch (err) {
        // Silent — health is optional
    }
}

// ============================================================
// Rendering — Dashboard
// ============================================================

function renderAgents(agents) {
    const container = document.getElementById('agentList');
    if (!agents || agents.length === 0) {
        container.innerHTML = '<p class="empty-state">No agents running</p>';
        return;
    }

    container.innerHTML = agents.map(agent => {
        const stats = agent.stats || {};
        return `
        <div class="agent-card" data-status="${agent.status}">
            <div class="agent-info">
                <span class="agent-name">${ROLE_ICONS[agent.role] || '🤖'} ${agent.role}</span>
                <span class="agent-model">${agent.model}</span>
                <span class="agent-stats-line">msgs: ${stats.messages_received || 0} | llm: ${stats.llm_calls || 0} | ${formatSeconds(stats.total_think_seconds)}</span>
            </div>
            <span class="agent-status status-${agent.status}">${agent.status}</span>
        </div>`;
    }).join('');
}

function renderProjects(projects) {
    const container = document.getElementById('projectsList');
    if (!projects || projects.length === 0) {
        container.innerHTML = '<p class="empty-state">No projects yet. Submit an idea above!</p>';
        return;
    }

    container.innerHTML = projects.map(project => {
        const created = new Date(project.created_at).toLocaleDateString();
        const idea = project.idea || project.description || '';
        return `
        <div class="project-card" onclick="viewProject('${project.id}')">
            <div class="project-info">
                <h3>${escapeHtml(project.name || 'Untitled Project')}</h3>
                <p>${escapeHtml(idea.substring(0, 120))}</p>
                <div class="project-meta">
                    <span>Created ${created}</span>
                    <span>Tokens: ${formatNumber(project.tokens_used || 0)}</span>
                </div>
            </div>
            <span class="project-status-pill" data-status="${project.status}">${project.status}</span>
        </div>`;
    }).join('');
}

function viewProject(projectId) {
    currentProjectId = projectId;

    // Show and activate project tab
    const tab = document.getElementById('tabProject');
    tab.style.display = '';
    switchView('project');
}

// ============================================================
// Rendering — Project Detail
// ============================================================

function renderProjectDetail(data) {
    const project = data.project;
    const progress = data.progress;
    currentProjectStatus = project.status;

    document.getElementById('tabProjectLabel').textContent = project.name || 'Project';
    document.getElementById('projectDetailName').textContent = project.name || 'Untitled Project';
    document.getElementById('projectDetailIdea').textContent = project.idea || project.description;

    // Status badge
    const badge = document.getElementById('projectDetailStatus');
    badge.textContent = project.status;
    badge.setAttribute('data-status', project.status);

    // Pause/Resume button
    const btn = document.getElementById('btnPause');
    if (project.status === 'active') {
        btn.textContent = '⏸ Pause';
        btn.className = 'btn-sm btn-warning';
        btn.style.display = '';
    } else if (project.status === 'paused') {
        btn.textContent = '▶ Resume';
        btn.className = 'btn-sm btn-success';
        btn.style.display = '';
    } else {
        btn.style.display = 'none';
    }

    // Progress
    if (progress) {
        const pct = progress.progress_pct || 0;
        document.getElementById('projectProgressBar').style.width = `${pct}%`;
        document.getElementById('projectProgressStats').textContent =
            `${progress.completed || 0} / ${progress.total_tasks || 0} tasks complete (${pct}%)`;

        const tu = progress.token_usage;
        if (tu) {
            document.getElementById('tokenBudget').textContent =
                `Tokens: ${formatNumber(tu.tokens_used)} / ${formatNumber(tu.budget)} (${tu.pct_used}% used)`;
        }
    }
}

// ============================================================
// Rendering — Kanban Board
// ============================================================

const KANBAN_STATE_MAP = {
    pending: 'Pending', assigned: 'Assigned', in_progress: 'InProgress',
    review: 'Review', revision: 'Review', approved: 'Done', done: 'Done', failed: 'Failed',
};

function renderKanban(tasks) {
    // Clear all columns
    ['Pending', 'Assigned', 'InProgress', 'Review', 'Done', 'Failed'].forEach(col => {
        document.getElementById(`kanban${col}`).innerHTML = '';
    });

    const counts = { Pending: 0, Assigned: 0, InProgress: 0, Review: 0, Done: 0, Failed: 0 };

    tasks.forEach(task => {
        const col = KANBAN_STATE_MAP[task.state] || 'Pending';
        counts[col]++;

        const agentIcon = ROLE_ICONS[task.assigned_to] || '❓';
        const card = document.createElement('div');
        card.className = 'kanban-card';
        card.onclick = () => showTaskModal(task);
        card.innerHTML = `
            <div class="card-title">${escapeHtml(task.title)}</div>
            <div class="card-meta">
                <span class="card-agent">${agentIcon} ${task.assigned_to || '—'}</span>
                <span class="card-iteration">${task.iteration}/${task.max_iterations}</span>
            </div>`;

        document.getElementById(`kanban${col}`).appendChild(card);
    });

    // Update counts
    document.getElementById('countPending').textContent = counts.Pending;
    document.getElementById('countAssigned').textContent = counts.Assigned;
    document.getElementById('countInProgress').textContent = counts.InProgress;
    document.getElementById('countReview').textContent = counts.Review;
    document.getElementById('countDone').textContent = counts.Done;
    document.getElementById('countFailed').textContent = counts.Failed;
}

function showTaskModal(task) {
    document.getElementById('taskModalTitle').textContent = task.title;

    const body = document.getElementById('taskModalBody');
    body.innerHTML = `
        <div class="task-detail-grid">
            <span class="label">State</span>
            <span class="value"><span class="badge">${task.state}</span></span>
            <span class="label">Assigned to</span>
            <span class="value">${ROLE_ICONS[task.assigned_to] || '—'} ${task.assigned_to || 'unassigned'}</span>
            <span class="label">Priority</span>
            <span class="value">${task.priority}</span>
            <span class="label">Iteration</span>
            <span class="value">${task.iteration} / ${task.max_iterations}</span>
            <span class="label">Tokens</span>
            <span class="value">${formatNumber(task.tokens_used || 0)}</span>
            <span class="label">Updated</span>
            <span class="value">${new Date(task.updated_at).toLocaleString()}</span>
        </div>
        <h4 style="margin-bottom:0.4rem;font-size:0.85rem;">Description</h4>
        <div class="task-description">${escapeHtml(task.description)}</div>
        ${task.feedback ? `
        <h4 style="margin:0.6rem 0 0.4rem;font-size:0.85rem;">Last Feedback</h4>
        <div class="task-description">${escapeHtml(task.feedback)}</div>` : ''}
        ${task.deliverable ? `
        <h4 style="margin:0.6rem 0 0.4rem;font-size:0.85rem;">Deliverable</h4>
        <div class="task-description">${escapeHtml(task.deliverable)}</div>` : ''}
    `;

    const footer = document.getElementById('taskModalFooter');
    footer.innerHTML = '';
    if (task.state === 'failed') {
        const retryBtn = document.createElement('button');
        retryBtn.className = 'btn-sm btn-success';
        retryBtn.textContent = '🔄 Retry Task';
        retryBtn.onclick = () => retryTask(task.id);
        footer.appendChild(retryBtn);
    }

    document.getElementById('taskModal').classList.add('open');
}

// ============================================================
// Rendering — Conversation Viewer
// ============================================================

function renderConversation(messages) {
    const container = document.getElementById('conversationViewer');
    if (!messages || messages.length === 0) {
        container.innerHTML = '<p class="empty-state">No messages yet</p>';
        return;
    }

    container.innerHTML = messages.map(msg => {
        const sender = String(msg.sender || 'system');
        const recipient = String(msg.recipient || '');
        const type = String(msg.type || '');
        const payload = typeof msg.payload === 'string' ? JSON.parse(msg.payload) : (msg.payload || {});
        const content = payload.content || '';
        const time = msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : '';

        return `
        <div class="msg-bubble" data-sender="${sender}">
            <div class="msg-header">
                <span class="msg-sender">${ROLE_ICONS[sender] || '⚙️'} ${sender} → ${ROLE_ICONS[recipient] || '🤖'} ${recipient}</span>
                <span class="msg-meta">
                    <span class="activity-type">${TYPE_LABELS[type] || type}</span>
                    <span>${time}</span>
                </span>
            </div>
            <div class="msg-body">${escapeHtml(content)}</div>
        </div>`;
    }).join('');
}

// ============================================================
// Rendering — Artifact Browser
// ============================================================

function renderArtifacts(artifacts) {
    const container = document.getElementById('artifactBrowser');
    if (!artifacts || artifacts.length === 0) {
        container.innerHTML = '<p class="empty-state">No artifacts generated yet</p>';
        return;
    }

    container.innerHTML = artifacts.map(a => {
        const ext = (a.filename || '').split('.').pop().toLowerCase();
        const icon = FILE_ICONS[ext] || '📄';
        const agentIcon = ROLE_ICONS[a.agent] || '🤖';
        const size = a.content ? `${(a.content.length / 1024).toFixed(1)} KB` : '—';
        const time = a.created_at ? new Date(a.created_at).toLocaleString() : '';

        return `
        <div class="artifact-row" onclick='showArtifactModal(${JSON.stringify(escapeHtml(a.filename))}, ${JSON.stringify(a.id)})'>
            <div class="artifact-file-info">
                <span class="artifact-icon">${icon}</span>
                <span class="artifact-filename">${escapeHtml(a.filename)}</span>
            </div>
            <div class="artifact-meta">
                <span>${agentIcon} ${a.agent}</span>
                <span>v${a.version}</span>
                <span>${a.artifact_type}</span>
                <span>${size}</span>
                <span>${time}</span>
            </div>
        </div>`;
    }).join('');
}

async function showArtifactModal(filename, artifactId) {
    document.getElementById('artifactModalTitle').textContent = filename;
    document.getElementById('artifactModalCode').textContent = 'Loading...';
    document.getElementById('artifactModal').classList.add('open');

    try {
        // Fetch artifact content — we already have it in the list, but let's use
        // the project artifacts endpoint which returns full content.
        const artifacts = await apiFetch(`/api/projects/${currentProjectId}/artifacts`);
        const artifact = artifacts.find(a => a.id === artifactId);
        if (artifact) {
            document.getElementById('artifactModalCode').textContent = artifact.content;
        } else {
            document.getElementById('artifactModalCode').textContent = '(Artifact not found)';
        }
    } catch (err) {
        document.getElementById('artifactModalCode').textContent = `Error: ${err.message}`;
    }
}

// ============================================================
// Activity Feed
// ============================================================

function addActivityItem(data) {
    const container = document.getElementById('activityFeed');

    // Remove empty state
    const emptyState = container.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    const sender = String(data.sender || 'system');
    const recipient = String(data.recipient || '');
    const type = String(data.type || '');
    const content = data.payload?.content || JSON.stringify(data.data || data);
    const time = data.timestamp
        ? new Date(data.timestamp).toLocaleTimeString()
        : new Date().toLocaleTimeString();

    const item = document.createElement('div');
    item.className = 'activity-item';
    item.setAttribute('data-sender', sender);
    item.innerHTML = `
        <div class="activity-header">
            <span class="activity-agents">
                ${ROLE_ICONS[sender] || '⚙️'} ${sender}
                ${recipient ? `→ ${ROLE_ICONS[recipient] || '🤖'} ${recipient}` : ''}
            </span>
            <span style="display:flex;gap:0.4rem;align-items:center;">
                <span class="activity-type">${TYPE_LABELS[type] || type}</span>
                <span class="activity-time">${time}</span>
            </span>
        </div>
        <div class="activity-content">${escapeHtml(content.substring(0, 400))}${content.length > 400 ? '…' : ''}</div>
    `;

    container.insertBefore(item, container.firstChild);

    // Apply current filter
    applyFeedFilter(item);

    // Cap feed size
    while (container.children.length > 200) {
        container.removeChild(container.lastChild);
    }
}

function filterFeed() {
    document.querySelectorAll('#activityFeed .activity-item').forEach(applyFeedFilter);
}

function applyFeedFilter(item) {
    const filter = document.getElementById('feedFilter').value;
    if (filter === 'all') {
        item.classList.remove('filtered');
    } else {
        const sender = item.getAttribute('data-sender');
        item.classList.toggle('filtered', sender !== filter);
    }
}

function clearFeed() {
    document.getElementById('activityFeed').innerHTML =
        '<p class="empty-state">Waiting for activity...</p>';
}

// ============================================================
// Modal helpers
// ============================================================

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('open');
}

// Close modals on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal-overlay.open').forEach(m => m.classList.remove('open'));
    }
});

// ============================================================
// Utilities
// ============================================================

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function formatNumber(n) {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
    return String(n);
}

function formatSeconds(s) {
    if (!s) return '0s';
    if (s < 60) return `${Math.round(s)}s`;
    if (s < 3600) return `${Math.round(s / 60)}m`;
    return `${(s / 3600).toFixed(1)}h`;
}

// ============================================================
// Initialize
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    loadAgents();
    loadProjects();
    loadHealth();

    // Periodic refresh
    setInterval(loadAgents, 10_000);
    setInterval(loadProjects, 30_000);
    setInterval(loadHealth, 15_000);

    // Refresh project detail if viewing one
    setInterval(() => {
        if (currentProjectId && document.getElementById('viewProject').classList.contains('active')) {
            loadProjectDetail(currentProjectId);
        }
    }, 10_000);
});
