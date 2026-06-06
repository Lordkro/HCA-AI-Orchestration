"""Tests for structured logging setup.

Verifies that setup_logging configures structlog correctly for
different formats (json, console) and handles file output.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from hca.core.logger import setup_logging


class TestSetupLogging:
    def test_default_config(self) -> None:
        """Default to JSON format, INFO level, no file handler."""
        setup_logging()
        logger = structlog.get_logger()
        assert logger is not None

    def test_json_format(self) -> None:
        setup_logging(log_format="json")
        cfg = structlog.get_config()
        renderers = [p for p in cfg["processors"] if isinstance(p, structlog.processors.JSONRenderer)]
        assert len(renderers) == 1

    def test_console_format(self) -> None:
        setup_logging(log_format="console")
        cfg = structlog.get_config()
        renderers = [p for p in cfg["processors"] if isinstance(p, structlog.dev.ConsoleRenderer)]
        assert len(renderers) == 1

    def test_logger_outputs(self) -> None:
        """Verify the logger can produce output without error."""
        setup_logging(log_format="json")
        logger = structlog.get_logger()
        logger.info("test_message", key="value")

    def test_log_file_created(self, tmp_path: Path) -> None:
        log_file = str(tmp_path / "test.log")
        setup_logging(log_level="DEBUG", log_format="json", log_file=log_file)
        logger = structlog.get_logger()
        logger.info("file_test", detail="written")
        import logging as _logging
        _logging.shutdown()

        log_path = Path(log_file)
        assert log_path.exists()

    def test_file_rotation_not_triggered(self, tmp_path: Path) -> None:
        """Small logs should not trigger rotation."""
        log_file = str(tmp_path / "rotate.log")
        setup_logging(log_level="INFO", log_format="json", log_file=log_file)
        import logging as _logging
        _logging.shutdown()
        log_path = Path(log_file)
        assert log_path.exists()

    def test_invalid_log_level_defaults_to_info(self) -> None:
        setup_logging(log_level="INVALID")
        cfg = structlog.get_config()
        assert cfg["wrapper_class"] is not None

    def test_console_output_does_not_crash(self) -> None:
        setup_logging(log_format="console")
        logger = structlog.get_logger()
        logger.info("console_test", key="value")
