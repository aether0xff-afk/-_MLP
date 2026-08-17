"""Numerical contracts for scale-independent coefficient normalization."""

from __future__ import annotations

import numpy as np

from cubic_mlp.cardano import normalize_cubic_coefficients


def test_normalization_is_invariant_to_equation_scaling() -> None:
    coefficients = np.array([2.0, 1.0, -25.0, 12.0])
    expected_normalized, expected_variable_scale = normalize_cubic_coefficients(
        coefficients
    )

    for equation_scale in (1e-100, 1e-8, -7.5, 1e8, -1e100):
        normalized, variable_scale = normalize_cubic_coefficients(
            equation_scale * coefficients
        )
        np.testing.assert_allclose(
            normalized,
            expected_normalized,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            variable_scale,
            expected_variable_scale,
            rtol=1e-13,
            atol=1e-13,
        )


def test_normalized_coefficients_are_bounded_and_round_trip_to_monic_form() -> None:
    coefficients = np.array(
        [
            [1.0, 1e6, -1e-3, 7.0],
            [2.0, -3.0, 5e8, -1e12],
            [-5.0, 0.0, 0.0, 0.125],
            [0.25, -8.0, 2.0, -64.0],
        ]
    )

    normalized, variable_scale = normalize_cubic_coefficients(coefficients)

    assert normalized.shape == (4, 3)
    assert variable_scale.shape == (4,)
    assert np.all(np.isfinite(normalized))
    assert np.all(np.isfinite(variable_scale))
    assert np.all(variable_scale > 0.0)
    assert np.max(np.abs(normalized)) <= 1.0 + 8 * np.finfo(float).eps
    np.testing.assert_allclose(
        np.max(np.abs(normalized), axis=1),
        1.0,
        rtol=1e-13,
        atol=1e-13,
    )

    powers = np.column_stack(
        [variable_scale, variable_scale**2, variable_scale**3]
    )
    reconstructed_monic_tail = normalized * powers
    expected_monic_tail = coefficients[:, 1:] / coefficients[:, :1]
    np.testing.assert_allclose(
        reconstructed_monic_tail,
        expected_monic_tail,
        rtol=1e-13,
        atol=1e-13,
    )


def test_normalization_handles_zero_tail_and_extremely_small_variable_scales() -> None:
    zero_normalized, zero_scale = normalize_cubic_coefficients([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(zero_normalized, [0.0, 0.0, 0.0])
    assert zero_scale == 1.0

    quadratic_scale, scale_c = normalize_cubic_coefficients([1.0, 0.0, 1e-300, 0.0])
    linear_scale, scale_b = normalize_cubic_coefficients([1.0, 1e-300, 0.0, 0.0])

    np.testing.assert_array_equal(quadratic_scale, [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(linear_scale, [1.0, 0.0, 0.0])
    assert scale_c == 1e-150
    assert scale_b == 1e-300


def test_normalization_accepts_large_but_finite_monic_coefficient_ratios() -> None:
    normalized, variable_scale = normalize_cubic_coefficients(
        [1.0, -3e12, 2e24, -1e36]
    )

    assert np.all(np.isfinite(normalized))
    assert np.isfinite(variable_scale)
    assert np.max(np.abs(normalized)) <= 1.0 + 1e-14
