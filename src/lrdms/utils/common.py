from typing import Any, Callable
from functools import partial


def exists(x: Any) -> bool:
    return x is not None


def default(x: Any, default: Any) -> Any:
    return x if exists(x) else default


def named_partial(func: Callable, name: str = "", suffix: str = "", **kwargs) -> Callable:
    f = partial(func, **kwargs)
    if name:
        f.__name__ = name
    else:
        f.__name__ = func.__name__ + suffix
    return f
