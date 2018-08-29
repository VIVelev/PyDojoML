from .base import BaseError

__all__ = [
    "MethodNotSupportedError",
    "ParameterError",
]

class MethodNotSupportedError(BaseError):
    pass

class ParameterError(BaseError):
    pass
