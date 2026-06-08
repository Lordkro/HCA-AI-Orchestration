"""Tests for tool definitions and validation logic in hca.core.tools."""

from __future__ import annotations

import pytest

from hca.core.tools import (
    CREATE_TASK_TOOL,
    FETCH_PAGE_TOOL,
    INSTALL_PACKAGE_TOOL,
    LIST_FILES_TOOL,
    READ_FILE_TOOL,
    SUBMIT_REVIEW_TOOL,
    WEB_SEARCH_TOOL,
    WRITE_FILE_TOOL,
    format_validation_errors,
    validate_and_log,
    validate_tool_call,
)

# ============================================================
# Tool definition structure tests
# ============================================================

ALL_TOOLS: list[tuple[str, dict, list[str]]] = [
    ("create_task", CREATE_TASK_TOOL, ["title", "description", "assigned_to"]),
    ("write_file", WRITE_FILE_TOOL, ["path", "content", "artifact_type"]),
    ("submit_review", SUBMIT_REVIEW_TOOL, ["verdict", "summary"]),
    ("web_search", WEB_SEARCH_TOOL, ["query"]),
    ("fetch_page", FETCH_PAGE_TOOL, ["url"]),
    ("install_package", INSTALL_PACKAGE_TOOL, ["packages"]),
    ("list_files", LIST_FILES_TOOL, ["path"]),
    ("read_file", READ_FILE_TOOL, ["path"]),
]


class TestToolDefinitions:
    """Verify every tool definition has the correct structure."""

    @pytest.mark.parametrize("name,tool_def,required", ALL_TOOLS)
    def test_tool_structure(self, name: str, tool_def: dict, required: list[str]) -> None:
        assert tool_def["type"] == "function"
        fn = tool_def["function"]
        assert fn["name"] == name
        assert isinstance(fn["description"], str)
        assert len(fn["description"]) > 5
        params = fn["parameters"]
        assert params["type"] == "object"
        for field in required:
            assert field in params["properties"], f"{name}: missing required property {field}"
        assert params.get("required", []) == required, (
            f"{name}: required fields mismatch {params.get('required')} != {required}"
        )

    @pytest.mark.parametrize("name,tool_def,_", ALL_TOOLS)
    def test_tool_descriptions_are_descriptive(self, name: str, tool_def: dict, _) -> None:
        desc = tool_def["function"]["description"]
        assert len(desc) > 10, f"{name}: description too short"


# ============================================================
# validate_tool_call tests
# ============================================================


class TestValidateToolCall:
    """Tests for validate_tool_call against each tool definition."""

    @pytest.mark.parametrize(
        "tool_def,valid_call",
        [
            (CREATE_TASK_TOOL, {"name": "create_task", "arguments": {"title": "T", "description": "D", "assigned_to": "coder"}}),
            (WRITE_FILE_TOOL, {"name": "write_file", "arguments": {"path": "f.py", "content": "x=1", "artifact_type": "code"}}),
            (SUBMIT_REVIEW_TOOL, {"name": "submit_review", "arguments": {"verdict": "approved", "summary": "Good."}}),
            (WEB_SEARCH_TOOL, {"name": "web_search", "arguments": {"query": "python async"}}),
            (FETCH_PAGE_TOOL, {"name": "fetch_page", "arguments": {"url": "https://example.com"}}),
            (INSTALL_PACKAGE_TOOL, {"name": "install_package", "arguments": {"packages": ["requests"]}}),
            (LIST_FILES_TOOL, {"name": "list_files", "arguments": {"path": "."}}),
            (READ_FILE_TOOL, {"name": "read_file", "arguments": {"path": "main.py"}}),
        ],
    )
    def test_valid_call(self, tool_def: dict, valid_call: dict) -> None:
        errors = validate_tool_call(valid_call, tool_def)
        assert errors == []

    @pytest.mark.parametrize(
        "tool_def,invalid_call,expected_error_substring",
        [
            (CREATE_TASK_TOOL, {"name": "create_task", "arguments": {"title": "T"}}, "missing required field 'description'"),
            (CREATE_TASK_TOOL, {"name": "create_task", "arguments": {"title": "T", "description": "D", "assigned_to": "invalid_role"}}, "invalid value 'invalid_role'"),
            (WRITE_FILE_TOOL, {"name": "write_file", "arguments": {"path": "f.py", "content": "x=1"}}, "missing required field 'artifact_type'"),
            (WRITE_FILE_TOOL, {"name": "write_file", "arguments": {"path": "f.py", "content": "x=1", "artifact_type": "invalid_type"}}, "invalid value 'invalid_type'"),
            (SUBMIT_REVIEW_TOOL, {"name": "submit_review", "arguments": {"verdict": "maybe"}}, "invalid value 'maybe'"),
            (WEB_SEARCH_TOOL, {"name": "web_search", "arguments": {}}, "missing required field 'query'"),
            (FETCH_PAGE_TOOL, {"name": "fetch_page", "arguments": {}}, "missing required field 'url'"),
            (INSTALL_PACKAGE_TOOL, {"name": "install_package", "arguments": {"packages": "requests"}}, "must be an array"),
            (LIST_FILES_TOOL, {"name": "list_files", "arguments": {}}, "missing required field 'path'"),
            (READ_FILE_TOOL, {"name": "read_file", "arguments": {}}, "missing required field 'path'"),
        ],
    )
    def test_invalid_call(self, tool_def: dict, invalid_call: dict, expected_error_substring: str) -> None:
        errors = validate_tool_call(invalid_call, tool_def)
        assert any(expected_error_substring in e for e in errors), (
            f"Expected error containing {expected_error_substring!r}, got {errors}"
        )

    def test_non_dict_arguments(self) -> None:
        errors = validate_tool_call(
            {"name": "test", "arguments": "not_a_dict"},
            CREATE_TASK_TOOL,
        )
        assert len(errors) == 1
        assert "must be an object" in errors[0]

    def test_missing_arguments_key(self) -> None:
        errors = validate_tool_call({"name": "test"}, WEB_SEARCH_TOOL)
        # Missing arguments defaults to {} which triggers missing required field
        assert len(errors) >= 1


# ============================================================
# validate_and_log tests
# ============================================================


class TestValidateAndLog:
    """Tests for the batch validate_and_log function."""

    def test_all_valid(self) -> None:
        calls = [
            {"name": "web_search", "arguments": {"query": "python"}},
            {"name": "fetch_page", "arguments": {"url": "https://example.com"}},
        ]
        tool_defs = [WEB_SEARCH_TOOL, FETCH_PAGE_TOOL]
        valid, errors = validate_and_log(calls, tool_defs, agent_name="test")
        assert len(valid) == 2
        assert errors == []

    def test_unknown_tool(self) -> None:
        calls = [{"name": "nonexistent_tool", "arguments": {}}]
        valid, errors = validate_and_log(calls, [CREATE_TASK_TOOL], agent_name="test")
        assert valid == []
        assert len(errors) == 1
        assert "Unknown tool" in errors[0]

    def test_mixed_valid_invalid(self) -> None:
        calls = [
            {"name": "web_search", "arguments": {"query": "python"}},
            {"name": "web_search", "arguments": {}},
        ]
        valid, errors = validate_and_log(calls, [WEB_SEARCH_TOOL], agent_name="test")
        assert len(valid) == 1
        assert len(errors) >= 1


# ============================================================
# format_validation_errors tests
# ============================================================


class TestFormatValidationErrors:
    def test_empty(self) -> None:
        assert format_validation_errors([]) == ""

    def test_single_error(self) -> None:
        result = format_validation_errors(["Something is wrong"])
        assert "Something is wrong" in result
        assert "corrected tool call" in result

    def test_multiple_errors(self) -> None:
        result = format_validation_errors(["Error 1", "Error 2"])
        assert "Error 1" in result
        assert "Error 2" in result
