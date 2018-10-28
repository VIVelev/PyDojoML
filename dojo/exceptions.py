from .base import BaseError

__all__ = [
    "MethodNotSupportedError",
    "ParameterError",
    "InvalidProblemError",
]


class MethodNotSupportedError(BaseError):
    pass

class ParameterError(BaseError):
    pass

class InvalidProblemError(BaseError):
    pass
