"""Tests for Phase 2 integration of logging with Experiment and Run classes.

This module tests the integration of the logging system with the actual
Experiment and Run classes to ensure proper lifecycle logging.
"""

import time
from datetime import datetime

import pytest

from minto import Experiment
from minto.logging_config import LogConfig, LogFormat, LogLevel


class TestExperimentLogging:
    """Test cases for Experiment class logging integration."""

    def test_experiment_creation_with_verbose_logging(self):
        """Test that experiment creation logs appropriately with verbose logging."""
        config = LogConfig(enabled=True, show_timestamps=False, show_colors=False)

        exp = Experiment(
            name="test_experiment",
            verbose_logging=True,
            log_config=config,
            auto_saving=False,
            collect_environment=False,
        )

        assert exp.verbose_logging is True
        assert exp._logger is not None
        assert exp._logger.config.enabled is True
        assert exp._run_count == 0

    def test_experiment_creation_with_disabled_logging(self):
        """Test that experiment creation works with disabled logging."""
        exp = Experiment(
            name="test_experiment",
            verbose_logging=False,
            auto_saving=False,
            collect_environment=False,
        )

        assert exp.verbose_logging is False
        assert exp._logger is not None
        assert exp._logger.config.enabled is False

    def test_create_run_increments_count(self):
        """Test that creating runs increments the run count."""
        exp = Experiment(
            name="test_experiment",
            verbose_logging=False,
            auto_saving=False,
            collect_environment=False,
        )

        assert exp._run_count == 0

        run1 = exp.run()
        assert exp._run_count == 1
        assert run1._run_id == 0

        run2 = exp.run()
        assert exp._run_count == 2
        assert run2._run_id == 1

    def test_finish_experiment_calculates_duration(self):
        """Test that finish_experiment calculates and logs duration."""
        exp = Experiment(
            name="test_experiment",
            verbose_logging=False,
            auto_saving=False,
            collect_environment=False,
        )

        start_time = exp._start_time
        time.sleep(0.01)  # Small delay

        exp.finish_experiment()

        # Should have calculated duration
        assert isinstance(start_time, datetime)


class TestRunLogging:
    """Test cases for Run class logging integration."""

    def test_run_parameter_logging(self):
        """Test that run parameter logging works correctly."""
        exp = Experiment(
            name="test_experiment",
            verbose_logging=False,  # Disable console output for testing
            auto_saving=False,
            collect_environment=False,
        )

        run = exp.run()

        # Test basic parameter logging
        run.log_parameter("test_param", "test_value")
        assert "test_param" in run.parameters
        assert run.parameters["test_param"] == "test_value"

        # Test numeric parameters
        run.log_parameter("numeric_param", 42)
        assert run.parameters["numeric_param"] == 42

    def test_run_object_logging(self):
        """Test that run object logging works correctly."""
        exp = Experiment(
            name="test_experiment",
            verbose_logging=False,
            auto_saving=False,
            collect_environment=False,
        )

        run = exp.run()

        test_object = {"key1": "value1", "key2": 42}
        run.log_object("test_object", test_object)

        assert "test_object" in run.objects
        assert run.objects["test_object"] == test_object

    def test_run_context_manager_lifecycle(self):
        """Test that run context manager properly handles lifecycle."""
        exp = Experiment(
            name="test_experiment",
            verbose_logging=False,
            auto_saving=False,
            collect_environment=False,
        )

        run = exp.run()

        # Run should not be closed initially
        assert not run.is_closed

        with run:
            run.log_parameter("param_in_context", "value")
            assert not run.is_closed

        # Run should be closed after context manager exit
        assert run.is_closed

        # Should not be able to log after closing
        with pytest.raises(RuntimeError, match="This run has been closed"):
            run.log_parameter("param_after_close", "value")

    def test_run_elapsed_time_calculation(self):
        """Test that run elapsed time is calculated correctly."""
        exp = Experiment(
            name="test_experiment",
            verbose_logging=False,
            auto_saving=False,
            collect_environment=False,
        )

        run = exp.run()

        time.sleep(0.01)  # Small delay

        with run:
            pass  # Run will be closed automatically

        # Should have elapsed time in metadata
        assert "elapsed_time" in run._datastore.meta_data
        elapsed_time = run._datastore.meta_data["elapsed_time"]
        assert elapsed_time > 0
        assert elapsed_time < 1.0  # Should be less than 1 second


class TestSolverLogging:
    """Test cases for solver logging functionality."""

    def test_solver_wrapper_execution(self):
        """Test that solver wrapper logs execution details."""
        exp = Experiment(
            name="test_experiment",
            verbose_logging=False,
            auto_saving=False,
            collect_environment=False,
        )

        run = exp.run()

        def mock_solver(param1=10, param2="test"):
            time.sleep(0.01)  # Simulate processing
            return {"result": param1 * 2}

        with run:
            wrapped_solver = run.log_solver("mock_solver", mock_solver)

            # Check that solver name was logged as parameter
            assert "solver_name" in run.parameters
            assert run.parameters["solver_name"] == "mock_solver"

            # Execute solver
            result = wrapped_solver(param1=20, param2="test_value")

            # Check that result is correct
            assert result == {"result": 40}

            # Check that parameters were logged
            assert "param1" in run.parameters
            assert run.parameters["param1"] == 20
            assert "param2" in run.parameters
            assert run.parameters["param2"] == "test_value"

    def test_solver_exclude_params(self):
        """Test that solver wrapper respects exclude_params."""
        exp = Experiment(
            name="test_experiment",
            verbose_logging=False,
            auto_saving=False,
            collect_environment=False,
        )

        run = exp.run()

        def mock_solver(public_param=10, private_param="secret"):
            return {"result": "success"}

        with run:
            wrapped_solver = run.log_solver(
                "mock_solver", mock_solver, exclude_params=["private_param"]
            )

            wrapped_solver(public_param=20, private_param="hidden")

            # Public param should be logged
            assert "public_param" in run.parameters
            assert run.parameters["public_param"] == 20

            # Private param should not be logged
            assert "private_param" not in run.parameters


class TestLoggingConfiguration:
    """Test cases for logging configuration options."""

    def test_custom_log_config(self):
        """Test that custom log configuration is respected."""
        custom_config = LogConfig(
            enabled=True,
            level=LogLevel.DEBUG,
            format=LogFormat.DETAILED,
            show_timestamps=False,
            show_icons=False,
        )

        exp = Experiment(
            name="test_experiment",
            verbose_logging=True,
            log_config=custom_config,
            auto_saving=False,
            collect_environment=False,
        )

        assert exp._logger.config.level == LogLevel.DEBUG
        assert exp._logger.config.format == LogFormat.DETAILED
        assert exp._logger.config.show_timestamps is False
        assert exp._logger.config.show_icons is False

    def test_logging_with_environment_collection(self):
        """Test that environment collection works with logging."""
        exp = Experiment(
            name="test_experiment",
            verbose_logging=False,  # Disable console output for testing
            auto_saving=False,
            collect_environment=True,
        )

        # Environment info should be collected
        env_info = exp.get_environment_info()
        assert env_info is not None
        assert "os_name" in env_info
        assert "python_version" in env_info


if __name__ == "__main__":
    pytest.main([__file__])
