from typing import Any


def exists(x: Any) -> bool:
    return x is not None


def default(x: Any, default: Any) -> Any:
    return x if exists(x) else default
