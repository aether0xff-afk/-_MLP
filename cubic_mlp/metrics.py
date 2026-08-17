"""근 집합 오차와 다항식 backward residual 평가."""

from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

PERMUTATIONS = np.asarray(tuple(permutations(range(3))), dtype=np.int64)


def pairs_to_complex(values: ArrayLike) -> NDArray[np.complex128]:
    array = np.asarray(values)
    if array.shape[-2:] != (3, 2):
        raise ValueError("마지막 두 차원은 (3, 2)여야 합니다.")
    return array[..., 0].astype(np.float64) + 1j * array[..., 1].astype(np.float64)


def complex_to_pairs(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.complex128)
    return np.stack([array.real, array.imag], axis=-1)


def optimally_match_roots(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """각 샘플에서 6개 순열을 전수 조사해 target을 prediction에 맞춘다."""

    predicted = np.asarray(predictions, dtype=np.complex128)
    target = np.asarray(targets, dtype=np.complex128)
    if predicted.ndim != 2 or predicted.shape[1] != 3 or target.shape != predicted.shape:
        raise ValueError("predictions와 targets는 (N, 3)이어야 합니다.")
    candidates = target[:, PERMUTATIONS]
    squared = np.abs(predicted[:, None, :] - candidates) ** 2
    best = squared.mean(axis=2).argmin(axis=1)
    matched = candidates[np.arange(len(target)), best]
    errors = np.abs(predicted - matched)
    return matched, errors


def normalized_residuals(
    normalized_coefficients: ArrayLike,
    roots: ArrayLike,
) -> NDArray[np.float64]:
    coefficients = np.asarray(normalized_coefficients, dtype=np.float64)
    values = np.asarray(roots, dtype=np.complex128)
    b = coefficients[:, 0, None]
    c = coefficients[:, 1, None]
    d = coefficients[:, 2, None]
    polynomial = values**3 + b * values**2 + c * values + d
    absolute = np.abs(values)
    denominator = absolute**3 + np.abs(b) * absolute**2 + np.abs(c) * absolute + np.abs(d) + 1e-12
    return np.abs(polynomial) / denominator


def _summary(values: NDArray[np.float64]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def evaluate_predictions(
    normalized_coefficients: ArrayLike,
    predicted_roots: ArrayLike,
    target_roots: ArrayLike,
    *,
    regimes: ArrayLike | None = None,
) -> dict[str, Any]:
    predicted = np.asarray(predicted_roots, dtype=np.complex128)
    targets = np.asarray(target_roots, dtype=np.complex128)
    _, errors = optimally_match_roots(predicted, targets)
    residual = normalized_residuals(normalized_coefficients, predicted)
    sample_max = errors.max(axis=1)

    result: dict[str, Any] = {
        "sample_count": int(len(predicted)),
        "root_mae": float(errors.mean()),
        "root_rmse": float(np.sqrt(np.mean(errors**2))),
        "sample_max_error": _summary(sample_max),
        "normalized_residual": _summary(residual.reshape(-1)),
    }
    if regimes is not None:
        regime_array = np.asarray(regimes)
        by_regime: dict[str, Any] = {}
        for name in np.unique(regime_array):
            mask = regime_array == name
            by_regime[str(name)] = {
                "sample_count": int(mask.sum()),
                "root_mae": float(errors[mask].mean()),
                "root_rmse": float(np.sqrt(np.mean(errors[mask] ** 2))),
                "sample_max_error": _summary(sample_max[mask]),
                "normalized_residual": _summary(residual[mask].reshape(-1)),
            }
        result["by_generation_regime"] = by_regime
    return result

