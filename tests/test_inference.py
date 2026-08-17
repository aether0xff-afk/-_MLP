"""Tests for deterministic post-processing applied to MLP predictions."""

from __future__ import annotations

import numpy as np

from cubic_mlp.inference import (
    CubicMLPSolver,
    batch_vieta_set_errors,
    enforce_real_cubic_structure,
    enforce_real_cubic_structure_batch,
    newton_polish_roots,
    newton_polish_roots_batch_scaled,
    polynomial_residuals,
    vieta_set_error,
)
from cubic_mlp.cardano import solve_cubic_cardano_batch
from cubic_mlp.checkpoint import save_checkpoint
from cubic_mlp.model import CubicRootMLP, ModelConfig


def test_structure_cleanup_makes_one_real_root_and_a_conjugate_pair() -> None:
    # z^3 + z - 10 has roots 2 and -1 +/- 2j.
    noisy_predictions = np.array(
        [-1.2 + 1.7j, 2.0 + 0.1j, -0.8 - 2.3j],
        dtype=np.complex128,
    )

    structured = enforce_real_cubic_structure(
        noisy_predictions,
        [0.0, 1.0, -10.0],
    )

    np.testing.assert_allclose(
        structured,
        [-1.0 - 2.0j, -1.0 + 2.0j, 2.0 + 0.0j],
        rtol=0.0,
        atol=1e-12,
    )


def test_structure_cleanup_projects_three_real_case_onto_real_axis() -> None:
    # z^3 - 6z^2 + 11z - 6 has three real roots.
    noisy_predictions = np.array(
        [3.0 + 0.15j, 1.0 - 0.20j, 2.0 + 0.05j],
        dtype=np.complex128,
    )

    structured = enforce_real_cubic_structure(
        noisy_predictions,
        [-6.0, 11.0, -6.0],
    )

    np.testing.assert_allclose(
        structured,
        [1.0, 2.0, 3.0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        noisy_predictions,
        [3.0 + 0.15j, 1.0 - 0.20j, 2.0 + 0.05j],
    )


def test_structure_cleanup_keeps_small_complex_pair() -> None:
    # Delta=5.787e-10 is small in absolute terms but unambiguously positive.
    exact_roots = np.array([0.0, 0.05j, -0.05j], dtype=np.complex128)

    structured = enforce_real_cubic_structure(exact_roots, [0.0, 0.0025, 0.0])

    np.testing.assert_allclose(
        structured,
        [-0.05j, 0.0, 0.05j],
        rtol=0.0,
        atol=1e-14,
    )


def test_vectorized_structure_cleanup_matches_scalar_path() -> None:
    roots = np.array(
        [
            [-1.2 + 1.7j, 2.0 + 0.1j, -0.8 - 2.3j],
            [3.0 + 0.15j, 1.0 - 0.20j, 2.0 + 0.05j],
        ]
    )
    normalized_coefficients = np.array(
        [[0.0, 1.0, -10.0], [-6.0, 11.0, -6.0]]
    )
    expected = np.stack(
        [
            enforce_real_cubic_structure(row_roots, row_coefficients)
            for row_roots, row_coefficients in zip(
                roots, normalized_coefficients, strict=True
            )
        ]
    )

    actual = enforce_real_cubic_structure_batch(roots, normalized_coefficients)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_newton_polishing_strictly_improves_residuals_for_nearby_simple_roots() -> None:
    coefficients = np.array([1.0, -6.0, 11.0, -6.0])
    approximate_roots = np.array([0.85, 2.15, 3.10], dtype=np.complex128)
    before_absolute, before_relative = polynomial_residuals(
        coefficients,
        approximate_roots,
    )

    polished = newton_polish_roots(
        approximate_roots,
        coefficients,
        steps=2,
    )
    after_absolute, after_relative = polynomial_residuals(coefficients, polished)

    assert np.all(after_absolute < before_absolute)
    assert np.all(after_relative < before_relative)
    assert np.max(after_absolute) < 0.01 * np.max(before_absolute)
    np.testing.assert_array_equal(approximate_roots, [0.85, 2.15, 3.10])


def test_newton_and_normalized_residual_are_invariant_to_equation_scale() -> None:
    coefficients = np.array([1.0, -6.0, 11.0, -6.0])
    approximate_roots = np.array([0.9, 2.1, 3.1], dtype=np.complex128)
    expected = newton_polish_roots(approximate_roots, coefficients, steps=2)
    _, expected_normalized = polynomial_residuals(coefficients, expected)

    scaled_coefficients = coefficients * 1e-100
    actual = newton_polish_roots(approximate_roots, scaled_coefficients, steps=2)
    _, actual_normalized = polynomial_residuals(scaled_coefficients, actual)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        actual_normalized,
        expected_normalized,
        rtol=1e-10,
        atol=1e-16,
    )


def test_newton_backtracking_rejects_a_full_step_that_worsens_residual() -> None:
    coefficients = np.array([1.0, -6.0, 11.0, -6.0])
    approximate = np.array([1.225, 2.1, 3.1], dtype=np.complex128)
    before, _ = polynomial_residuals(coefficients, approximate)

    polished = newton_polish_roots(approximate, coefficients, steps=1)
    after, _ = polynomial_residuals(coefficients, polished)

    assert after[0] < before[0]


def test_vectorized_newton_polish_reduces_batch_residuals() -> None:
    coefficients = np.array(
        [[-6.0, 11.0, -6.0], [0.0, 1.0, -10.0]], dtype=np.float64
    )
    approximate = np.array(
        [
            [0.9, 2.1, 3.1],
            [2.1, -1.1 + 1.9j, -1.1 - 1.9j],
        ],
        dtype=np.complex128,
    )
    before = np.abs(
        approximate**3
        + coefficients[:, 0, None] * approximate**2
        + coefficients[:, 1, None] * approximate
        + coefficients[:, 2, None]
    )

    polished = newton_polish_roots_batch_scaled(
        approximate, coefficients, steps=3
    )
    after = np.abs(
        polished**3
        + coefficients[:, 0, None] * polished**2
        + coefficients[:, 1, None] * polished
        + coefficients[:, 2, None]
    )

    assert np.all(after < before)


def test_vieta_set_error_detects_duplicate_roots_and_is_scale_invariant() -> None:
    coefficients = np.array([1.0, -6.0, 11.0, -6.0])
    exact = vieta_set_error(coefficients, [1.0, 2.0, 3.0])
    duplicate = vieta_set_error(coefficients, [1.0, 1.0, 3.0])
    scaled = vieta_set_error(1e-100 * coefficients, [1.0, 2.0, 3.0])

    np.testing.assert_array_equal(exact, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(scaled, exact)
    assert np.max(duplicate) > 0.0


def test_batch_vieta_errors_match_single_equation_version() -> None:
    coefficients = np.array([[-6.0, 11.0, -6.0], [0.0, 1.0, -10.0]])
    roots = np.array([[1.0, 2.0, 3.0], [2.0, -1.0 + 2j, -1.0 - 2j]])

    errors = batch_vieta_set_errors(coefficients, roots)

    np.testing.assert_allclose(errors, 0.0, rtol=0.0, atol=1e-14)


def test_reusable_batch_solver_and_forced_hybrid_fallback(tmp_path) -> None:
    config = ModelConfig(hidden_sizes=(8, 6))
    model = CubicRootMLP(config)
    checkpoint = tmp_path / "batch.pt"
    save_checkpoint(
        checkpoint,
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "input_mean": [0.0, 0.0, 0.0],
            "input_std": [1.0, 1.0, 1.0],
        },
    )
    coefficients = np.array(
        [
            [1.0, -6.0, 11.0, -6.0],
            [1.0, 0.0, 1.0, -10.0],
            [1.0, 0.0, -3.0, 2.0],
        ]
    )
    solver = CubicMLPSolver(checkpoint, device="cpu")

    batch = solver.predict_batch(coefficients, batch_size=2)
    hybrid = solver.predict_hybrid(
        coefficients,
        batch_size=2,
        discriminant_relative_threshold=2.0,
    )

    assert batch.structured_roots.shape == (3, 3)
    assert np.all(np.isfinite(batch.structured_roots))
    assert np.all(hybrid.fallback_mask)
    np.testing.assert_allclose(
        hybrid.roots,
        solve_cubic_cardano_batch(coefficients),
        rtol=0.0,
        atol=1e-12,
    )
