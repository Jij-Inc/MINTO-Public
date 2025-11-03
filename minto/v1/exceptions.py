"""Exceptions for minto v1 module.

This module defines custom exceptions for the minto v1 module to provide
more informative error messages when data loading fails.
"""


class MintoError(Exception):
    """Base exception for all minto v1 errors."""

    pass


class DataLoadError(MintoError):
    """Base exception for errors that occur during data loading."""

    pass


class FileNotFoundError(DataLoadError):
    """Exception raised when a required file is not found."""

    pass


class FileCorruptionError(DataLoadError):
    """Exception raised when a file is corrupted or has invalid format."""

    pass


class MetadataError(DataLoadError):
    """Base exception for errors related to metadata."""

    pass


class MissingMetadataError(MetadataError):
    """Exception raised when required metadata is missing."""

    pass


class InvalidMetadataError(MetadataError):
    """Exception raised when metadata has invalid format or values."""

    pass
