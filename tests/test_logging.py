"""Tests for logging configuration and formatter.

This module tests the logging configuration, formatting, and
basic logger functionality.
"""

import pytest

from minto.logger import MintoLogger, configure_logging, get_logger
from minto.logging_config import LogConfig, LogFormat, LogFormatter, LogLevel


class TestLogConfig:
    """Test cases for LogConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LogConfig()
        assert config.enabled is True
        assert config.level == LogLevel.INFO
        assert config.format == LogFormat.SIMPLE
        assert config.show_timestamps is True
        assert config.show_icons is True
        assert config.show_colors is True
        assert config.indent_size == 2
        assert config.max_parameter_length == 100

    def test_should_log(self):
        """Test should_log method with different levels."""
        config = LogConfig(level=LogLevel.INFO)

        assert config.should_log(LogLevel.ERROR) is True
        assert config.should_log(LogLevel.WARNING) is True
        assert config.should_log(LogLevel.INFO) is True
        assert config.should_log(LogLevel.DEBUG) is False

        # Test with disabled logging
        config.enabled = False
        assert config.should_log(LogLevel.ERROR) is False

    def test_different_log_levels(self):
        """Test should_log with different configured levels."""
        # DEBUG level should allow all messages
        config_debug = LogConfig(level=LogLevel.DEBUG)
        assert all(config_debug.should_log(level) for level in LogLevel)

        # ERROR level should only allow error messages
        config_error = LogConfig(level=LogLevel.ERROR)
        assert config_error.should_log(LogLevel.ERROR) is True
        assert config_error.should_log(LogLevel.WARNING) is False
        assert config_error.should_log(LogLevel.INFO) is False
        assert config_error.should_log(LogLevel.DEBUG) is False


class TestLogFormatter:
    """Test cases for LogFormatter class."""

    def test_format_message_simple(self):
        """Test simple message formatting."""
        config = LogConfig(show_timestamps=False, show_icons=False, show_colors=False)
        formatter = LogFormatter(config)

        result = formatter.format_message(LogLevel.INFO, "Test message", indent_level=0)
        assert result == "Test message"

        result = formatter.format_message(LogLevel.INFO, "Test message", indent_level=1)
        assert result == "  ├─ Test message"

    def test_format_message_with_timestamps(self):
        """Test message formatting with timestamps."""
        config = LogConfig(show_timestamps=True, show_icons=False, show_colors=False)
        formatter = LogFormatter(config)

        result = formatter.format_message(LogLevel.INFO, "Test message", indent_level=0)
        # Should start with timestamp pattern [YYYY-MM-DD HH:MM:SS]
        assert result.startswith("[")
        assert "] Test message" in result

    def test_format_message_with_icons(self):
        """Test message formatting with icons."""
        config = LogConfig(show_timestamps=False, show_icons=True, show_colors=False)
        formatter = LogFormatter(config)

        result = formatter.format_message(
            LogLevel.INFO, "Test message", indent_level=0, icon_type="experiment_start"
        )
        assert "🚀" in result
        assert "Test message" in result

    def test_format_message_different_formats(self):
        """Test different format styles."""
        # Test MINIMAL format
        config_minimal = LogConfig(
            format=LogFormat.MINIMAL,
            show_timestamps=False,
            show_icons=False,
            show_colors=False,
        )
        formatter_minimal = LogFormatter(config_minimal)
        result = formatter_minimal.format_message(LogLevel.INFO, "Test message")
        assert result == "Test message"

        # Test DETAILED format
        config_detailed = LogConfig(
            format=LogFormat.DETAILED,
            show_timestamps=False,
            show_icons=False,
            show_colors=False,
        )
        formatter_detailed = LogFormatter(config_detailed)
        result = formatter_detailed.format_message(LogLevel.INFO, "Test message")
        assert "[INFO]" in result
        assert "Test message" in result

    def test_truncate_value(self):
        """Test value truncation."""
        config = LogConfig(max_parameter_length=10)
        formatter = LogFormatter(config)

        # Short value should not be truncated
        short_value = "short"
        assert formatter.truncate_value(short_value) == "short"

        # Long value should be truncated
        long_value = "this is a very long value that exceeds the limit"
        result = formatter.truncate_value(long_value)
        assert len(result) == 10  # 7 chars + "..."
        assert result.endswith("...")

    def test_disabled_logging(self):
        """Test that disabled logging returns empty strings."""
        config = LogConfig(enabled=False)
        formatter = LogFormatter(config)

        result = formatter.format_message(LogLevel.INFO, "Test message")
        assert result == ""


class TestMintoLogger:
    """Test cases for MintoLogger class."""

    def test_logger_creation(self):
        """Test logger creation with default config."""
        logger = MintoLogger()
        assert logger.config is not None
        assert logger.formatter is not None
        assert logger._current_indent_level == 0

    def test_logger_with_custom_config(self):
        """Test logger creation with custom config."""
        config = LogConfig(enabled=False)
        logger = MintoLogger(config)
        assert logger.config.enabled is False

    def test_indent_level_tracking(self):
        """Test that indent level is tracked correctly."""
        config = LogConfig(enabled=False)  # Disable output for testing
        logger = MintoLogger(config)

        # Initially at experiment level
        assert logger._current_indent_level == 0

        # Starting experiment should reset to 0
        logger.log_experiment_start("test")
        assert logger._current_indent_level == 0

        # Starting run should set to 1
        logger.log_run_start(0)
        assert logger._current_indent_level == 1

        # Ending run should reset to 0
        logger.log_run_end(0, 1.0)
        assert logger._current_indent_level == 0


class TestGlobalLogger:
    """Test cases for global logger functions."""

    def test_get_logger(self):
        """Test getting global logger instance."""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2  # Should be the same instance

    def test_configure_logging(self):
        """Test configure_logging function."""
        configure_logging(
            enabled=False,
            level=LogLevel.ERROR,
            show_timestamps=False,
            show_icons=False,
            show_colors=False,
        )

        logger = get_logger()
        assert logger.config.enabled is False
        assert logger.config.level == LogLevel.ERROR
        assert logger.config.show_timestamps is False
        assert logger.config.show_icons is False
        assert logger.config.show_colors is False


if __name__ == "__main__":
    pytest.main([__file__])
