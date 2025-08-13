"""Logger module for Minto experiment tracking.

This module provides a unified logging interface for experiment
and run operations with hierarchical output formatting.
"""

import sys
import typing as typ

from .logging_config import LogConfig, LogFormatter, LogLevel


class MintoLogger:
    """Main logger class for Minto experiments and runs.

    This class provides a unified interface for logging experiment
    and run activities with proper indentation and formatting.
    """

    def __init__(self, config: typ.Optional[LogConfig] = None):
        """Initialize the logger with configuration.

        Args:
            config: Logging configuration. If None, uses default config.
        """
        self.config = config or LogConfig()
        self.formatter = LogFormatter(self.config)
        self._current_indent_level = 0

    def _print(self, message: str) -> None:
        """Print message to stdout.

        Args:
            message: The message to print.
        """
        if message:  # Only print non-empty messages
            print(message, file=sys.stdout, flush=True)

    def log_experiment_start(self, experiment_name: str) -> None:
        """Log the start of an experiment.

        Args:
            experiment_name: Name of the experiment.
        """
        message = f"Starting experiment '{experiment_name}'"
        formatted = self.formatter.format_message(
            LogLevel.INFO, message, indent_level=0, icon_type="experiment_start"
        )
        self._print(formatted)
        self._current_indent_level = 0

    def log_experiment_end(
        self, experiment_name: str, duration: float, run_count: int
    ) -> None:
        """Log the end of an experiment.

        Args:
            experiment_name: Name of the experiment.
            duration: Total duration in seconds.
            run_count: Number of runs completed.
        """
        message = (
            f"Experiment '{experiment_name}' completed: {run_count} runs, "
            f"total time: {duration:.1f}s"
        )
        formatted = self.formatter.format_message(
            LogLevel.INFO, message, indent_level=0, icon_type="experiment_end"
        )
        self._print(formatted)
        self._current_indent_level = 0

    def log_environment_info(self, info: dict[str, typ.Any]) -> None:
        """Log environment information.

        Args:
            info: Dictionary containing environment information.
        """
        if not self.config.should_log(LogLevel.INFO):
            return

        # Log summary first
        summary = self.formatter.format_environment_summary(info)
        formatted = self.formatter.format_message(
            LogLevel.INFO,
            f"Environment: {summary}",
            indent_level=1,
            icon_type="environment",
        )
        self._print(formatted)

        # Log detailed info if not minimal format
        if self.config.format.value != "minimal":
            env_lines = self.formatter.format_environment_info(info, 1)
            for line in env_lines:
                if line:  # Skip empty lines
                    self._print(line)

    def log_run_start(self, run_id: int) -> None:
        """Log the start of a run.

        Args:
            run_id: ID of the run.
        """
        message = f"Created run #{run_id}"
        formatted = self.formatter.format_message(
            LogLevel.INFO, message, indent_level=1, icon_type="run_start"
        )
        self._print(formatted)
        self._current_indent_level = 1

    def log_run_end(self, run_id: int, duration: float) -> None:
        """Log the end of a run.

        Args:
            run_id: ID of the run.
            duration: Duration of the run in seconds.
        """
        message = f"Run #{run_id} completed ({duration:.1f}s)"
        formatted = self.formatter.format_message(
            LogLevel.INFO, message, indent_level=1, icon_type="run_end"
        )
        self._print(formatted)
        self._current_indent_level = 0

    def log_parameter(self, name: str, value: typ.Any) -> None:
        """Log a parameter setting.

        Args:
            name: Parameter name.
            value: Parameter value.
        """
        truncated_value = self.formatter.truncate_value(value)
        message = f"Parameter: {name} = {truncated_value}"
        formatted = self.formatter.format_message(
            LogLevel.INFO, message, indent_level=2, icon_type="parameter"
        )
        self._print(formatted)

    def log_solution(self, name: str, solution_info: str) -> None:
        """Log a solution.

        Args:
            name: Solution name.
            solution_info: Brief description of the solution.
        """
        message = f"Solution '{name}': {solution_info}"
        formatted = self.formatter.format_message(
            LogLevel.INFO, message, indent_level=2, icon_type="solution"
        )
        self._print(formatted)

    def log_sampleset(
        self, name: str, num_samples: int, min_energy: typ.Optional[float] = None
    ) -> None:
        """Log a sampleset.

        Args:
            name: Sampleset name.
            num_samples: Number of samples.
            min_energy: Minimum energy value (optional).
        """
        if min_energy is not None:
            message = (
                f"SampleSet '{name}': {num_samples} samples, "
                f"min_energy={min_energy:.3f}"
            )
        else:
            message = f"SampleSet '{name}': {num_samples} samples"

        formatted = self.formatter.format_message(
            LogLevel.INFO, message, indent_level=2, icon_type="sampleset"
        )
        self._print(formatted)

    def log_solver(
        self, solver_name: str, execution_time: typ.Optional[float] = None
    ) -> None:
        """Log solver execution.

        Args:
            solver_name: Name of the solver.
            execution_time: Execution time in seconds (optional).
        """
        if execution_time is not None:
            message = f"Solver '{solver_name}' executed ({execution_time:.3f}s)"
        else:
            message = f"Solver '{solver_name}' executed"

        formatted = self.formatter.format_message(
            LogLevel.INFO, message, indent_level=2, icon_type="solver"
        )
        self._print(formatted)

    def log_object(self, name: str, obj_type: str, size_info: str = "") -> None:
        """Log an object.

        Args:
            name: Object name.
            obj_type: Type of the object.
            size_info: Additional size/content information.
        """
        if size_info:
            message = f"Object '{name}' ({obj_type}): {size_info}"
        else:
            message = f"Object '{name}' ({obj_type})"

        formatted = self.formatter.format_message(
            LogLevel.INFO,
            message,
            indent_level=2,
            icon_type="parameter",  # Reuse parameter icon for objects
        )
        self._print(formatted)

    def log_warning(self, message: str) -> None:
        """Log a warning message.

        Args:
            message: Warning message.
        """
        formatted = self.formatter.format_message(
            LogLevel.WARNING,
            message,
            indent_level=self._current_indent_level,
            icon_type="warning",
        )
        self._print(formatted)

    def log_error(self, message: str) -> None:
        """Log an error message.

        Args:
            message: Error message.
        """
        formatted = self.formatter.format_message(
            LogLevel.ERROR,
            message,
            indent_level=self._current_indent_level,
            icon_type="error",
        )
        self._print(formatted)

    def log_debug(self, message: str) -> None:
        """Log a debug message.

        Args:
            message: Debug message.
        """
        formatted = self.formatter.format_message(
            LogLevel.DEBUG, message, indent_level=self._current_indent_level
        )
        self._print(formatted)


# Global logger instance for easy access
_global_logger: typ.Optional[MintoLogger] = None


def get_logger() -> MintoLogger:
    """Get the global logger instance.

    Returns:
        The global MintoLogger instance.
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = MintoLogger()
    return _global_logger


def set_logger_config(config: LogConfig) -> None:
    """Set the configuration for the global logger.

    Args:
        config: New logging configuration.
    """
    global _global_logger
    _global_logger = MintoLogger(config)


def configure_logging(
    enabled: bool = True,
    level: LogLevel = LogLevel.INFO,
    show_timestamps: bool = True,
    show_icons: bool = True,
    show_colors: bool = True,
) -> None:
    """Configure logging with simplified parameters.

    Args:
        enabled: Whether to enable logging.
        level: Minimum log level to display.
        show_timestamps: Whether to show timestamps.
        show_icons: Whether to show emoji icons.
        show_colors: Whether to show colored output.
    """
    config = LogConfig(
        enabled=enabled,
        level=level,
        show_timestamps=show_timestamps,
        show_icons=show_icons,
        show_colors=show_colors,
    )
    set_logger_config(config)
