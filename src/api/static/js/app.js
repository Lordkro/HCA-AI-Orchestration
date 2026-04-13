// ============================================================
// HCA Orchestration — Dashboard Frontend
// ============================================================

const API_BASE = window.location.origin;
let ws = null;

// --------------------------------------------------------
// WebSocket Connection
// --------------------------------------------------------

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

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
    };

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
    if (connected) {
        el.innerHTML = '<span class="status-dot connected"></span><span>Connected</span>';
    } else {
        el.innerHTML = '<span class="status-dot disconnected"></span><span>Disconnected</span>';
    }
}

function handleRealtimeEvent(data) {
    // Add to activity feed
    if (data.sender && data.payload) {
        addActivityItem(data);
    }

    // Refresh agent status periodically
    loadAgents();
}

// --------------------------------------------------------
// API Calls
// --------------------------------------------------------

async function submitIdea() {
    const ideaInput = document.getElementById('ideaInput');
    const nameInput = document.getElementById('projectName');
    const idea = ideaInput.value.trim();

    if (!idea) {
        alert('Please enter a product idea.');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/projects/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                idea: idea,
                name: nameInput.value.trim() || '',
            }),
        });
        const result = await response.json();
        console.log('Project created:', result);

        // Clear inputs
        ideaInput.value = '';
        nameInput.value = '';

        // Add a UI notification
        addActivityItem({
            sender: 'user',
            recipient: 'pm',
            type: 'system',
            payload: { content: `New project submitted: "${idea.substring(0, 100)}..."` },
            timestamp: new Date().toISOString(),
        });

        // Refresh projects
        loadProjects();
    } catch (err) {
        console.error('Failed to submit idea:', err);
        alert('Failed to submit idea. Is the server running?');
    }
}

async function loadAgents() {
    try {
        const response = await fetch(`${API_BASE}/api/agents/`);
        const agents = await response.json();
        renderAgents(agents);
    } catch (err) {
        console.error('Failed to load agents:', err);
    }
}

async function loadProjects() {
    try {
        const response = await fetch(`${API_BASE}/api/projects/`);
        const projects = await response.json();
        renderProjects(projects);
    } catch (err) {
        console.error('Failed to load projects:', err);
    }
}

// --------------------------------------------------------
// Rendering
// --------------------------------------------------------

const ROLE_ICONS = {
    pm: '📋',
    research: '🔍',
    spec: '📐',
    coder: '💻',
    critic: '🔎',
    user: '👤',
    system: '⚙️',
};

function renderAgents(agents) {
    const container = document.getElementById('agentList');
    if (!agents || agents.length === 0) {
        container.innerHTML = '<p class="empty-state">No agents running</p>';
        return;
    }

    container.innerHTML = agents.map(agent => `
        <div class="agent-card" data-status="${agent.status}">
            <div class="agent-info">
                <span class="agent-name">${ROLE_ICONS[agent.role] || '🤖'} ${agent.role}</span>
                <span class="agent-model">${agent.model}</span>
            </div>
            <span class="agent-status status-${agent.status}">${agent.status}</span>
        </div>
    `).join('');
}

function renderProjects(projects) {
    const container = document.getElementById('projectsList');
    if (!projects || projects.length === 0) {
        container.innerHTML = '<p class="empty-state">No projects yet. Submit an idea above!</p>';
        return;
    }

    container.innerHTML = projects.map(project => `
        <div class="project-card" onclick="viewProject('${project.id}')">
            <div class="project-info">
                <h3>${project.name || 'Untitled Project'}</h3>
                <p>${project.idea ? project.idea.substring(0, 120) + '...' : project.description}</p>
            </div>
            <span class="project-status">${project.status}</span>
        </div>
    `).join('');
}

function addActivityItem(data) {
    const container = document.getElementById('activityFeed');

    // Remove empty state message
    const emptyState = container.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    const sender = data.sender || 'system';
    const recipient = data.recipient || '';
    const content = data.payload?.content || JSON.stringify(data.data || data);
    const time = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();

    const item = document.createElement('div');
    item.className = 'activity-item';
    item.setAttribute('data-sender', sender);
    item.innerHTML = `
        <div class="activity-header">
            <span class="activity-agents">
                ${ROLE_ICONS[sender] || '🤖'} ${sender}
                ${recipient ? `→ ${ROLE_ICONS[recipient] || '🤖'} ${recipient}` : ''}
            </span>
            <span class="activity-time">${time}</span>
        </div>
        <div class="activity-content">${escapeHtml(content.substring(0, 500))}${content.length > 500 ? '...' : ''}</div>
    `;

    // Add to top of feed
    container.insertBefore(item, container.firstChild);

    // Keep feed size manageable
    while (container.children.length > 100) {
        container.removeChild(container.lastChild);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function viewProject(projectId) {
    // TODO: Navigate to project detail view
    console.log('View project:', projectId);
}

// --------------------------------------------------------
// Initialize
// --------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    loadAgents();
    loadProjects();

    // Refresh agents every 10 seconds
    setInterval(loadAgents, 10000);
    setInterval(loadProjects, 30000);
});
