import multiprocessing as mp
from collections.abc import Callable
from functools import wraps
from typing import Concatenate, ParamSpec, TypeVar


class ResourceUpdateFromChildProcessError(Exception):
    """Exception raised when a child process tries to update the configuration of the parent process."""

    def __init__(self):
        super().__init__(
            "Attempted to update the configuration from a child process; only the parent process can update it."
        )


P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


def parent_process_only[T, **P, R](func: Callable[Concatenate[T, P], R]) -> Callable[Concatenate[T, P], R]:
    """Decorator to ensure that a method can only be called from the parent process."""

    @wraps(func)
    def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
        if mp.parent_process() is not None:
            raise ResourceUpdateFromChildProcessError
        return func(self, *args, **kwargs)

    return wrapper
