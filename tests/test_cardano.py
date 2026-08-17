"""Contract tests for the deterministic cubic-equation baseline."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from cubic_mlp.cardano import (
    canonicalize_roots,
    solve_cubic_cardano,
    solve_cubic_cardano_batch,
)


def _as_three_complex(values: object) -> np.ndarray:
    roots = np.asarray(values, dtype=np.complex128)
    assert roots.shape == (3,)
    assert np.all(np.isfinite(roots.real))
    assert np.all(np.isfinite(roots.imag))
    return roots


def _best_root_error(actual: object, expected: object) -> float:
    """Return the best max error over all six root assignments."""

    actual_roots = _as_three_complex(actual)
    expected_roots = _as_three_complex(expected)
    return min(
        float(np.max(np.abs(actual_roots - expected_roots[list(order)])))
        for order in permutations(range(3))
    )


def _assert_roots_match(
    actual: object,
    expected: object,
    *,
    atol: float = 1e-9,
) -> None:
    assert _best_root_error(actual, expected) <= atol


@pytest.mark.parametrize(
    ("coefficients", "expected_roots", "atol"),
    [
        pytest.param(
            [1.0, -6.0, 11.0, -6.0],
            [1.0, 2.0, 3.0],
            1e-9,
            id="three-distinct-real-roots",
        ),
        pytest.param(
            [1.0, 0.0, 1.0, -10.0],
            [2.0, -1.0 + 2.0j, -1.0 - 2.0j],
            1e-9,
            id="one-real-and-conjugate-pair",
        ),
        pytest.param(
            [1.0, 0.0, -3.0, 2.0],
            [-2.0, 1.0, 1.0],
            1e-7,
            id="double-root",
        ),
        pytest.param(
            [1.0, -9.0, 27.0, -27.0],
            [3.0, 3.0, 3.0],
            1e-6,
            id="triple-root",
        ),
        pytest.param(
            [2.0, 1.0, -25.0, 12.0],
            [-4.0, 0.5, 3.0],
            1e-9,
            id="non-monic",
        ),
    ],
)
def test_solve_cubic_cardano_known_polynomials(
    coefficients: list[float],
    expected_roots: list[complex],
    atol: float,
) -> None:
    roots = solve_cubic_cardano(coefficients)

    _assert_roots_match(roots, expected_roots, atol=atol)

    # A root finder should satisfy the input polynomial as well as agree with
    # the expected values. Normalizing by the coefficient scale keeps this
    # assertion meaningful for non-monic equations.
    roots_array = _as_three_complex(roots)
    residual = np.polyval(np.asarray(coefficients, dtype=float), roots_array)
    coefficient_scale = max(1.0, float(np.max(np.abs(coefficients))))
    assert np.max(np.abs(residual)) / coefficient_scale <= max(1e-9, 10 * atol)


def test_solver_is_invariant_to_nonzero_equation_scaling() -> None:
    expected = [-4.0, 0.5, 3.0]

    for scale in (1e-6, -7.5, 1e6):
        roots = solve_cubic_cardano(scale * np.array([2.0, 1.0, -25.0, 12.0]))
        _assert_roots_match(roots, expected, atol=1e-8)


def test_batch_solver_matches_scalar_solver() -> None:
    coefficients = np.array(
        [
            [1.0, -6.0, 11.0, -6.0],
            [1.0, 0.0, 1.0, -10.0],
            [2.0, 1.0, -25.0, 12.0],
        ]
    )

    batch_roots = np.asarray(solve_cubic_cardano_batch(coefficients))

    assert batch_roots.shape == (3, 3)
    for row, row_coefficients in zip(batch_roots, coefficients, strict=True):
        _assert_roots_match(
            row,
            solve_cubic_cardano(row_coefficients),
            atol=1e-10,
        )


def test_solver_rejects_zero_leading_coefficient() -> None:
    with pytest.raises(ValueError):
        solve_cubic_cardano([0.0, 1.0, -3.0, 2.0])


@pytest.mark.parametrize(
    "coefficients",
    [
        [1.0, 0.0, 0.0, -1e-6],
        [1.0, -1.54852268e-2, -1.68455040e-4, 2.69766974e-7],
    ],
)
def test_small_but_distinct_roots_are_not_misclassified_as_repeated(
    coefficients: list[float],
) -> None:
    roots = solve_cubic_cardano(coefficients)
    residual = np.abs(np.polyval(coefficients, roots))
    denominator = sum(
        abs(coefficient) * np.abs(roots) ** power
        for coefficient, power in zip(coefficients, (3, 2, 1, 0), strict=True)
    )
    assert np.max(residual / np.maximum(denominator, 1e-300)) < 1e-9


def test_canonicalize_roots_has_documented_lexicographic_order() -> None:
    unordered = np.array([1.0 + 2.0j, 1.0 - 2.0j, -2.0 + 0.0j])
    expected = np.array([-2.0 + 0.0j, 1.0 - 2.0j, 1.0 + 2.0j])

    actual = _as_three_complex(canonicalize_roots(unordered))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_canonicalize_roots_is_permutation_invariant_and_idempotent() -> None:
    roots = np.array([-0.25 + 0.0j, 2.5 + 3.0j, 2.5 - 3.0j])
    expected = _as_three_complex(canonicalize_roots(roots))

    for order in permutations(range(3)):
        actual = _as_three_complex(canonicalize_roots(roots[list(order)]))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    twice = _as_three_complex(canonicalize_roots(expected))
    np.testing.assert_allclose(twice, expected, rtol=0.0, atol=0.0)
