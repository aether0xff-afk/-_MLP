"""삼차방정식의 세 근을 근사하는 MLP 패키지."""

from .cardano import (
    canonicalize_roots,
    solve_cubic_cardano,
    solve_cubic_cardano_batch,
)
from .model import CubicRootMLP

__all__ = [
    "CubicRootMLP",
    "canonicalize_roots",
    "solve_cubic_cardano",
    "solve_cubic_cardano_batch",
]

__version__ = "0.1.0"

