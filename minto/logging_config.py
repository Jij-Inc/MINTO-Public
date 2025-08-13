"""Logging configuration module for Minto.

This module provides logging configuration and management utilities
for experiment and run execution monitoring.
"""

import enum
import typing as typ
from dataclasses import dataclass
from datetime import datetime


class LogLevel(enum.Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LogFormat(enum.Enum):
    """Log format enumeration."""

    SIMPLE = "simple"
    DETAILED = "detailed"
    MINIMAL = "minimal"


@dataclass
class LogConfig:
    """Configuration class for logging settings.

    This class manages logging behavior including verbosity levels,
    output formats, and display options.
    """

    enabled: bool = True
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.SIMPLE
    show_timestamps: bool = True
    show_icons: bool = True
    show_colors: bool = True
    indent_size: int = 2
    max_parameter_length: int = 100

    def should_log(self, level: LogLevel) -> bool:
        """Check if a message with given level should be logged.

        Args:
            level: The log level to check.

        Returns:
            True if the message should be logged, False otherwise.
        """
        if not self.enabled:
            return False

        level_priorities = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
        }

        return level_priorities[level] >= level_priorities[self.level]


class LogFormatter:
    """Formatter for log messages.

    This class handles the formatting of log messages according to
    the specified configuration.
    """

    def __init__(self, config: LogConfig):
        """Initialize the formatter with configuration.

        Args:
            config: Logging configuration.
        """
        self.config = config

        # Color codes for terminal output
        self.colors = (
            {
                LogLevel.DEBUG: "\033[90m",  # Gray
                LogLevel.INFO: "\033[36m",  # Cyan
                LogLevel.WARNING: "\033[33m",  # Yellow
                LogLevel.ERROR: "\033[31m",  # Red
            }
            if config.show_colors
            else {}
        )

        self.reset_color = "\033[0m" if config.show_colors else ""

        # Icons for different message types
        self.icons = (
            {
                "experiment_start": "🚀",
                "experiment_end": "🎯",
                "run_start": "🏃",
                "run_end": "✅",
                "parameter": "📝",
                "solution": "🎯",
                "sampleset": "📊",
                "solver": "⚙️",
                "error": "❌",
                "warning": "⚠️",
                "environment": "📊",
            }
            if config.show_icons
            else {}
        )

    def format_message(
        self,
        level: LogLevel,
        message: str,
        indent_level: int = 0,
        icon_type: typ.Optional[str] = None,
    ) -> str:
        """Format a log message according to configuration.

        Args:
            level: Log level of the message.
            message: The message content.
            indent_level: Indentation level (0 = experiment, 1 = run, 2 = operation).
            icon_type: Type of icon to use (optional).

        Returns:
            Formatted message string.
        """
        if not self.config.should_log(level):
            return ""

        parts = []

        # Add timestamp
        if self.config.show_timestamps:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            parts.append(f"[{timestamp}]")

        # Add indentation
        indent = " " * (indent_level * self.config.indent_size)
        if indent_level == 1:
            indent += "├─ "
        elif indent_level > 1:
            indent += "  ├─ "

        # Add icon
        icon = ""
        if icon_type and icon_type in self.icons:
            icon = f"{self.icons[icon_type]} "

        # Add color
        color_start = self.colors.get(level, "")
        color_end = self.reset_color if color_start else ""

        # Combine parts
        formatted_parts = [p for p in parts if p]
        prefix = " ".join(formatted_parts)

        if self.config.format == LogFormat.MINIMAL:
            if prefix:
                return f"{prefix} {indent}{icon}{color_start}{message}{color_end}"
            else:
                return f"{indent}{icon}{color_start}{message}{color_end}"
        elif self.config.format == LogFormat.DETAILED:
            level_str = f"[{level.value}]"
            if prefix:
                return (
                    f"{prefix} {level_str} {indent}{icon}{color_start}"
                    f"{message}{color_end}"
                )
            else:
                return f"{level_str} {indent}{icon}{color_start}{message}{color_end}"
        else:  # SIMPLE
            if prefix:
                return f"{prefix} {indent}{icon}{color_start}{message}{color_end}"
            else:
                return f"{indent}{icon}{color_start}{message}{color_end}"

    def truncate_value(self, value: typ.Any) -> str:
        """Truncate a value for display if it's too long.

        Args:
            value: The value to truncate.

        Returns:
            String representation of the value, possibly truncated.
        """
        str_value = str(value)
        if len(str_value) <= self.config.max_parameter_length:
            return str_value

        truncated = str_value[: self.config.max_parameter_length - 3]
        return f"{truncated}..."

    def format_memory_size(self, bytes_size: int) -> str:
        """Format memory size in human-readable format.

        Args:
            bytes_size: Size in bytes.

        Returns:
            Formatted string (e.g., "8.0 GB").
        """
        if bytes_size == 0:
            return "Unknown"

        size_float = float(bytes_size)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_float < 1024.0:
                return f"{size_float:.1f} {unit}"
            size_float /= 1024.0
        return f"{size_float:.1f} PB"

    def format_environment_summary(self, env_info: dict[str, typ.Any]) -> str:
        """Format a brief summary of environment information.

        Args:
            env_info: Dictionary containing environment information.

        Returns:
            Formatted summary string.
        """
        os_info = (
            f"{env_info.get('os_name', 'Unknown')} {env_info.get('os_version', '')}"
        )
        cpu_info = (
            f"{env_info.get('cpu_info', 'Unknown')} "
            f"({env_info.get('cpu_count', 'Unknown')} cores)"
        )
        memory = self.format_memory_size(env_info.get("memory_total", 0))
        python_version = (
            env_info.get("python_version", "Unknown").split()[0]
            if env_info.get("python_version")
            else "Unknown"
        )

        return (
            f"OS: {os_info}, CPU: {cpu_info}, Memory: {memory}, "
            f"Python: {python_version}"
        )

    def format_environment_info(
        self, env_info: dict[str, typ.Any], indent_level: int = 1
    ) -> list[str]:
        """Format full environment details for logging.

        Args:
            env_info: Dictionary containing environment information.
            indent_level: Indentation level for formatting.

        Returns:
            List of formatted strings, one per line.
        """
        lines = []

        # Header
        lines.append(
            self.format_message(
                LogLevel.INFO,
                "Environment Information",
                indent_level,
                "environment",
            )
        )

        # OS Information
        lines.append(
            self.format_message(
                LogLevel.INFO,
                f"OS: {env_info.get('os_name', 'Unknown')} "
                f"{env_info.get('os_version', '')}",
                indent_level + 1,
            )
        )
        lines.append(
            self.format_message(
                LogLevel.INFO,
                f"Platform: {env_info.get('platform_info', 'Unknown')}",
                indent_level + 1,
            )
        )

        # Hardware Information
        lines.append(
            self.format_message(
                LogLevel.INFO,
                f"CPU: {env_info.get('cpu_info', 'Unknown')} "
                f"({env_info.get('cpu_count', 'Unknown')} cores)",
                indent_level + 1,
            )
        )
        lines.append(
            self.format_message(
                LogLevel.INFO,
                f"Memory: {self.format_memory_size(env_info.get('memory_total', 0))}",
                indent_level + 1,
            )
        )
        lines.append(
            self.format_message(
                LogLevel.INFO,
                f"Architecture: {env_info.get('architecture', 'Unknown')}",
                indent_level + 1,
            )
        )

        # Python Environment
        python_version = (
            env_info.get("python_version", "Unknown").split()[0]
            if env_info.get("python_version")
            else "Unknown"
        )
        lines.append(
            self.format_message(
                LogLevel.INFO, f"Python: {python_version}", indent_level + 1
            )
        )

        if env_info.get("virtual_env"):
            lines.append(
                self.format_message(
                    LogLevel.INFO,
                    f"Virtual Environment: {env_info.get('virtual_env')}",
                    indent_level + 1,
                )
            )

        # Package Versions (optional)
        if env_info.get("package_versions") and self.config.format != LogFormat.MINIMAL:
            lines.append(
                self.format_message(
                    LogLevel.INFO, "Key Package Versions:", indent_level + 1
                )
            )
            for pkg, version in env_info.get("package_versions", {}).items():
                lines.append(
                    self.format_message(
                        LogLevel.DEBUG, f"{pkg}: {version}", indent_level + 2
                    )
                )

        return lines
