"""Pytest configuration for minto tests."""

import os
import warnings

import pytest


@pytest.fixture(autouse=True)
def disable_logging_in_tests(monkeypatch):
    """Automatically disable verbose logging for all tests unless explicitly enabled."""
    # Set environment variable to indicate we're in test mode
    monkeypatch.setenv("MINTO_TESTING", "true")


# Configure default test settings
def pytest_configure(config):
    """Set up test configuration."""
    # Suppress verbose output from minto during tests
    os.environ.setdefault("MINTO_LOG_LEVEL", "ERROR")

    # Filter out expected deprecation warnings from minto in tests
    warnings.filterwarnings(
        "ignore",
        message=".*outside run context is deprecated.*",
        category=DeprecationWarning,
    )
