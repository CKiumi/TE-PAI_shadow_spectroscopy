"""Backend-independent circuit IR and pluggable simulation backends.

Example
-------
    from pai_shadow.backend import Circuit, get_backend
    c = Circuit(2)
    c.h(0).rzz(0, 1, 0.5)
    be = get_backend("qulacs")
    print(be.expectation(c, "ZZ"))
"""

from __future__ import annotations

from .base import Backend, NoiseSpec
from .circuit import Circuit, Gate


def get_backend(name: str, noise: NoiseSpec | None = None) -> Backend:
    """Instantiate a backend by name ('qulacs')."""
    key = name.lower()
    if key == "qulacs":
        from .qulacs_backend import QulacsBackend
        return QulacsBackend(noise=noise)
    raise ValueError(f"Unknown backend {name!r}; expected 'qulacs'.")


__all__ = ["Backend", "NoiseSpec", "Circuit", "Gate", "get_backend"]
