"""Tool/function definitions for agent LLM tool calling.

Each tool is a dict in the Ollama/OpenAI function-calling format:
https://github.com/ollama/ollama/blob/main/docs/openai.md#tool-calls
"""

from __future__ import annotations

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


WEB_SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information on a given query, returning relevant snippets and links. Use this to investigate technologies, find documentation, and gather context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (use keywords, not full sentences)",
                },
            },
            "required": ["query"],
        },
    },
}

FETCH_PAGE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": "Fetch and read the full content of a webpage or documentation URL. Use this after web_search to get detailed information from a specific link.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Complete URL to fetch (must include scheme, e.g. https://example.com)",
                },
            },
            "required": ["url"],
        },
    },
}

INSTALL_PACKAGE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "install_package",
        "description": "Install one or more Python packages via pip. Use this when the project requires external dependencies.",
        "parameters": {
            "type": "object",
            "properties": {
                "packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Package names (optionally with version specifiers, e.g. requests>=2.31.0)",
                },
                "upgrade": {
                    "type": "boolean",
                    "description": "Whether to upgrade existing packages (--upgrade flag)",
                },
            },
            "required": ["packages"],
        },
    },
}

LIST_FILES_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files and directories in a workspace path. Use this to explore the project structure or verify file locations.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to project root (e.g. . or src or src/hca)",
                },
            },
            "required": ["path"],
        },
    },
}

READ_FILE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the full contents of a file in the project workspace. Use this to inspect existing code during revisions.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to project root (e.g. src/hca/main.py)",
                },
            },
            "required": ["path"],
        },
    },
}

# ======================================================================
# Tool Call Validation
# ======================================================================


def _extract_schema(tool_def: dict) -> dict | None:
    """Extract the JSON schema from a tool definition dict."""
    try:
        return tool_def["function"]["parameters"]
    except (KeyError, TypeError):
        return None


def validate_tool_call(tool_call: dict, tool_def: dict) -> list[str]:
    """Validate a tool call against its tool definition schema.

    Args:
        tool_call: The tool call dict with ``{"name": str, "arguments": dict}``.
        tool_def: The tool definition dict (one of the module-level constants).

    Returns:
        A list of human-readable validation errors.  Empty list = valid.
    """
    errors: list[str] = []
    name = tool_call.get("name", "?")
    args = tool_call.get("arguments", {})

    if not isinstance(args, dict):
        errors.append(f"Tool '{name}': arguments must be an object, got {type(args).__name__}")
        return errors

    schema = _extract_schema(tool_def)
    if schema is None:
        return errors  # No schema to validate against

    props = schema.get("properties", {})
    required = schema.get("required", [])

    # Check required fields
    for field_name in required:
        if field_name not in args or args[field_name] is None or args[field_name] == "":
            errors.append(
                f"Tool '{name}': missing required field '{field_name}'"
            )

    # Check field types and enum values
    for field_name, field_value in args.items():
        prop = props.get(field_name)
        if prop is None:
            continue

        expected_type = prop.get("type", "")
        enum_values = prop.get("enum", [])

        if enum_values and field_value not in enum_values:
            errors.append(
                f"Tool '{name}': field '{field_name}' has invalid value "
                f"'{field_value}'.  Allowed: {', '.join(enum_values)}"
            )

        if expected_type == "string" and not isinstance(field_value, str):
            errors.append(
                f"Tool '{name}': field '{field_name}' must be a string, "
                f"got {type(field_value).__name__}"
            )

        if expected_type == "array":
            if not isinstance(field_value, list):
                errors.append(
                    f"Tool '{name}': field '{field_name}' must be an array, "
                    f"got {type(field_value).__name__}"
                )
            else:
                item_schema = prop.get("items", {})
                item_type = item_schema.get("type", "")
                if item_type == "string":
                    for i, item in enumerate(field_value):
                        if not isinstance(item, str):
                            errors.append(
                                f"Tool '{name}': field '{field_name}'[{i}] must be a string, "
                                f"got {type(item).__name__}"
                            )

    return errors


def format_validation_errors(errors: list[str]) -> str:
    """Format validation errors into a human-readable message for the LLM."""
    if not errors:
        return ""
    lines = [
        "Your tool call had the following validation errors. Please fix them and try again:",
        "",
    ]
    for err in errors:
        lines.append(f"- {err}")
    lines.append("")
    lines.append("Provide corrected tool call(s) only.")
    return "\n".join(lines)


def validate_and_log(
    tool_calls: list[dict],
    tool_defs: list[dict],
    *,
    agent_name: str = "",
) -> tuple[list[dict], list[str]]:
    """Validate a batch of tool calls against their definitions.

    Args:
        tool_calls: List of tool call dicts.
        tool_defs: List of tool definition dicts.
        agent_name: Agent name for logging.

    Returns:
        (valid_calls, all_errors) where valid_calls contains only calls
        that passed validation, and all_errors is a flat list of error strings.
    """
    import structlog

    logger = structlog.get_logger()
    valid: list[dict] = []
    all_errors: list[str] = []
    name_to_def = {td["function"]["name"]: td for td in tool_defs}

    for call in tool_calls:
        call_name = call.get("name", "")
        tool_def = name_to_def.get(call_name)
        if tool_def is None:
            err = f"Unknown tool '{call_name}'"
            all_errors.append(err)
            logger.warning("tool_call_unknown", agent=agent_name, tool=call_name)
            continue

        errors = validate_tool_call(call, tool_def)
        if errors:
            all_errors.extend(errors)
            logger.warning(
                "tool_call_invalid",
                agent=agent_name,
                tool=call_name,
                errors=errors,
            )
        else:
            valid.append(call)

    return valid, all_errors
