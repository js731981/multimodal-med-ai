"""Domain errors for the inference pipeline."""


class InvalidImageInputError(ValueError):
    """Raised when an image cannot be decoded or is not a supported type."""
