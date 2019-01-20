from .base import DojoError

__all__ = [
    "MethodNotSupportedError",
    "ParameterError",
    "InvalidProblemError",
]


class MethodNotSupportedError(DojoError):
    pass

class ParameterError(DojoError):
    pass

class InvalidProblemError(DojoError):
    pass
