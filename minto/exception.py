class MINTOError(Exception):
    """Base exception class for MINTO-related errors.

    This class serves as the base exception class for all custom exceptions
    raised within the MINTO package. All other MINTO-specific exceptions should
    inherit from this class.

    Examples:
        >>> raise MINTOError("An error occurred in MINTO")
        MINTOError: An error occurred in MINTO

    Note:
        This is a simple exception class that doesn't add any additional functionality
        beyond what's provided by the base Exception class. It's primarily used for
        categorizing MINTO-specific errors.
    """

    pass
