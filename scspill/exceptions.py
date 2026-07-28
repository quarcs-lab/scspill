"""Custom exception classes for the scspill library.

Mirrors the exception hierarchy of ``mlsynth`` so that a future merge of the
two libraries is a rename, not a redesign.
"""


class ScspillError(Exception):
    """Base class for all custom exceptions in the scspill library."""


class ScspillConfigError(ScspillError):
    """Exception raised for errors in configuration."""


class ScspillDataError(ScspillError):
    """Exception raised for errors related to input data."""


class ScspillEstimationError(ScspillError):
    """Exception raised for errors during the estimation process."""


class ScspillPlottingError(ScspillError):
    """Exception raised for errors during plot generation."""
