"""Tests for root/coefficient conversion and synthetic training data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

data = pytest.importorskip("cubic_mlp.data")


def _dataset_arrays(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Support the two conventional public dataset representations."""

    if isinstance(dataset, dict):
        try:
            return np.asarray(dataset["coefficients"]), np.asarray(dataset["roots"])
        except KeyError as error:
            pytest.fail(f"dataset mapping is missing public field {error.args[0]!r}")

    if isinstance(dataset, (tuple, list)) and len(dataset) == 2:
        return np.asarray(dataset[0]), np.asarray(dataset[1])

    if hasattr(dataset, "coefficients") and hasattr(dataset, "roots"):
        return np.asarray(dataset.coefficients), np.asarray(dataset.roots)

    pytest.fail(
        "generate_dataset must return (coefficients, roots), a mapping with "
        "those keys, or an object exposing those attributes"
    )


@pytest.mark.skipif(
    not hasattr(data, "coefficients_from_roots"),
    reason="optional coefficients_from_roots API is not present",
)
def test_coefficients_from_roots_returns_monic_real_polynomial() -> None:
    roots = np.array([2.0, -1.0 + 2.0j, -1.0 - 2.0j])

    coefficients = np.asarray(data.coefficients_from_roots(roots))

    assert coefficients.shape == (4,)
    np.testing.assert_allclose(coefficients.imag, 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        coefficients.real,
        [1.0, 0.0, 1.0, -10.0],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.skipif(
    not hasattr(data, "coefficients_from_roots"),
    reason="optional coefficients_from_roots API is not present",
)
def test_roots_to_coefficients_to_cardano_round_trip() -> None:
    from cubic_mlp.cardano import solve_cubic_cardano

    roots = np.array([-4.0, 0.5, 3.0])
    coefficients = data.coefficients_from_roots(roots)
    recovered = np.asarray(solve_cubic_cardano(coefficients), dtype=np.complex128)

    np.testing.assert_allclose(
        np.sort_complex(recovered),
        np.sort_complex(roots.astype(np.complex128)),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.skipif(
    not hasattr(data, "generate_dataset"),
    reason="optional generate_dataset API is not present",
)
def test_generate_dataset_is_seed_reproducible_and_finite() -> None:
    first_coefficients, first_roots = _dataset_arrays(
        data.generate_dataset(32, seed=2026)
    )
    second_coefficients, second_roots = _dataset_arrays(
        data.generate_dataset(32, seed=2026)
    )

    assert first_coefficients.shape[0] == 32
    assert first_roots.shape[0] == 32
    assert first_coefficients.shape == second_coefficients.shape
    assert first_roots.shape == second_roots.shape
    assert np.all(np.isfinite(first_coefficients))
    assert np.all(np.isfinite(first_roots))
    np.testing.assert_array_equal(first_coefficients, second_coefficients)
    np.testing.assert_array_equal(first_roots, second_roots)


@pytest.mark.skipif(
    not hasattr(data, "generate_dataset"),
    reason="optional generate_dataset API is not present",
)
def test_generate_dataset_changes_with_seed() -> None:
    first_coefficients, first_roots = _dataset_arrays(data.generate_dataset(16, seed=1))
    second_coefficients, second_roots = _dataset_arrays(data.generate_dataset(16, seed=2))

    same_coefficients = np.array_equal(first_coefficients, second_coefficients)
    same_roots = np.array_equal(first_roots, second_roots)
    assert not (same_coefficients and same_roots)
