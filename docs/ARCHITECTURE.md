# Architecture

## Overview

The HCA (Hybrid Cognitive Architecture) is an autonomous AI development team powered by local LLMs via Ollama.

## System Components

### Agents
- **Project Manager Agent** - Orchestration and task management
- **Research Agent** - Technology investigation and feasibility analysis
- **Specification Agent** - Technical specification and architecture design
- **Coder Agent** - Implementation and code generation
- **Critic Agent** - Quality review and validation

### Core Subsystems

- **API Layer** - FastAPI-based REST and WebSocket APIs
- **Message Bus** - Inter-agent communication
- **Database** - SQLite for persistence
- **Ollama Client** - LLM interaction via Ollama
- **Orchestrator** - Task pipeline and guardrails
- **Logger** - Structured logging

## Directory Structure

```
src/hca/
├── agents/          # Agent implementations
├── api/             # API routes and handlers
├── core/            # Core services and utilities
├── orchestrator/    # Pipeline and task management
└── prompts/         # Agent system prompts
```

## Data Flow

1. User submits project idea via API
2. PM Agent breaks down into tasks
3. Tasks are distributed to appropriate agents
4. Results are validated by Critic Agent
5. Final output delivered to user
