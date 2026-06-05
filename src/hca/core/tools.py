"""Tool/function definitions for agent LLM tool calling.

Each tool is a dict in the Ollama/OpenAI function-calling format:
https://github.com/ollama/ollama/blob/main/docs/openai.md#tool-calls
"""

CREATE_TASK_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "create_task",
        "description": "Create a new task for an agent to work on in the project",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short descriptive title of the task",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the work to be done",
                },
                "assigned_to": {
                    "type": "string",
                    "enum": ["research", "spec", "coder"],
                    "description": "Which agent role should perform this task",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "critical"],
                    "description": "Priority level of the task",
                },
                "depends_on_titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Titles of tasks this depends on (use exact titles from other TASK blocks, empty list if none)",
                },
            },
            "required": ["title", "description", "assigned_to"],
        },
    },
}

WRITE_FILE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file to the project workspace. Use this for every file you create.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to project root (e.g. src/auth/login.py)",
                },
                "content": {
                    "type": "string",
                    "description": "Complete file contents",
                },
                "artifact_type": {
                    "type": "string",
                    "enum": ["code", "test", "doc", "config"],
                    "description": "Type of artifact being created",
                },
            },
            "required": ["path", "content", "artifact_type"],
        },
    },
}

SUBMIT_REVIEW_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "Submit a quality review verdict for a deliverable. Call this ONCE with your final decision.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["approved", "needs_revision"],
                    "description": "Whether the deliverable is approved or needs revision",
                },
                "summary": {
                    "type": "string",
                    "description": "One-paragraph overall assessment of the deliverable",
                },
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "severity": {
                                "type": "string",
                                "enum": ["critical", "major", "minor"],
                            },
                            "description": {"type": "string"},
                            "suggestion": {"type": "string"},
                        },
                        "required": ["severity", "description"],
                    },
                    "description": "List of specific issues found (empty if approved with no issues)",
                },
                "recommendations": {
                    "type": "string",
                    "description": "Suggestions for improvement (even if approved)",
                },
            },
            "required": ["verdict", "summary"],
        },
    },
}
